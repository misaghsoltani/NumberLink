"""Vectorized Gymnasium backend for the NumberLink environment.

This module provides a vectorized implementation of the NumberLink environment suitable for use with
:mod:`gymnasium.vector`. It exposes the :class:`NumberLinkRGBVectorEnv` class which runs multiple puzzle instances in
parallel and provides batched observations, actions, and information dictionaries.

See :func:`numberlink.level_setup.build_level_template` and :class:`numberlink.level_setup.LevelTemplate` for the level
construction utilities used by the environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING

from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np
from numpy.typing import NDArray

from .level_setup import build_level_template
from .number_render import build_endpoint_labels, render_bitmap_text_centered
from .types import ObsType, select_signed_dtype, select_unsigned_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, TypeAlias

    from gymnasium.core import RenderFrame

    from .config import GeneratorConfig, RenderConfig, RewardConfig, VariantConfig
    from .level_setup import LevelTemplate
    from .types import ActType, Coord, RenderMode, RGBInt

InfoValue: TypeAlias = NDArray[np.uint8] | NDArray[np.unsignedinteger] | NDArray[np.bool_] | list[str | None]
InfoDict: TypeAlias = dict[str, InfoValue]


LANE_NORMAL: np.uint8 = np.uint8(0)
LANE_VERTICAL: np.uint8 = np.uint8(1)
LANE_HORIZONTAL: np.uint8 = np.uint8(2)
LANE_BOTH: np.uint8 = np.uint8(3)


class NumberLinkRGBVectorEnv(VectorEnv[ObsType, NDArray[np.integer], NDArray[np.float32 | np.bool_]]):
    """Vectorized NumPy environment for NumberLink puzzles.

    Run multiple NumberLink instances in parallel and present batched observations and actions compatible with
    :mod:`gymnasium.vector`.

    The environment has two action modes. Use :attr:`variant.cell_switching_mode` to enable the cell switching mode.

    :ivar num_envs: Number of parallel environments.
    :vartype num_envs: int
    :ivar single_observation_space: Observation space for a single environment as a :class:`gymnasium.spaces.Box`.
    :ivar single_action_space: Action space for a single environment as a :class:`gymnasium.spaces.Discrete` object.
    :ivar observation_space: Batched observation space for all environments.
    :vartype observation_space: gymnasium.spaces.Space
    :ivar action_space: Batched action space for all environments.
    :vartype action_space: gymnasium.spaces.Space
    :ivar variant: Game rules and interaction mode configuration.
    :vartype variant: numberlink.config.VariantConfig
    :ivar level_id: Identifier for the current puzzle configuration or ``None`` when generated procedurally.
    :vartype level_id: str | None
    :ivar max_steps: Maximum number of steps before truncation.
    :vartype max_steps: int
    """

    metadata: dict[str, list[str] | AutoresetMode] = {
        "render_modes": ["rgb_array"],
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int,
        *,
        grid: Sequence[str] | None = None,
        render_mode: RenderMode | None = None,
        level_id: str | None = None,
        variant: VariantConfig | None = None,
        bridges: Iterable[Coord] | None = None,
        generator: GeneratorConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_config: RenderConfig | None = None,
        step_limit: int | None = None,
        palette: dict[str, RGBInt] | None = None,
        solution: list[list[Coord]] | None = None,
    ) -> None:
        """Initialize the vectorized environment and allocate state arrays.

        Build a :class:`numberlink.level_setup.LevelTemplate` using :func:`numberlink.level_setup.build_level_template`
        and allocate the batched NumPy arrays used to represent grid codes, lane arrays, stacks, heads, masks, and
        lookup tables.

        :param num_envs: Number of parallel environments to run.
        :param grid: Optional grid specification as a sequence of strings.
        :param render_mode: Rendering mode. Use ``'rgb_array'`` for image observations or ``None`` for no rendering.
        :param level_id: Optional identifier for a predefined level.
        :param variant: Configuration for game rules and interaction modes.
        :param bridges: Iterable of cell coordinates that allow crossing paths.
            See :class:`numberlink.level_setup.LevelTemplate`.
        :param generator: Procedural generation configuration.
        :param reward_config: Reward shaping parameters.
        :param render_config: Visual rendering style configuration.
        :param step_limit: Maximum steps before truncation. If ``None`` use ``10 * grid area``.
        :param palette: Optional mapping from color names to RGB tuples.
        :param solution: Optional list of per-color coordinate paths representing a solved puzzle.
        """
        self.render_mode = render_mode
        self._num_envs: int = int(num_envs)
        if self._num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self._source_grid: tuple[str, ...] | None = tuple(str(row) for row in grid) if grid is not None else None
        self._source_bridges: tuple[Coord, ...] | None = (
            tuple((coord[0], coord[1]) for coord in bridges) if bridges is not None else None
        )
        self._source_level_id: str | None = level_id
        self._variant_override: VariantConfig | None = variant
        self._reward_config_override: RewardConfig | None = reward_config
        self._render_config_override: RenderConfig | None = render_config
        self._palette_override: dict[str, RGBInt] | None = dict(palette) if palette is not None else None
        generator_normalized: GeneratorConfig | None = self._normalize_generator_config(generator)
        self._generator_config: GeneratorConfig | None = generator_normalized
        self._step_limit_override: int | None = step_limit
        self._source_solution: list[list[Coord]] | None = (
            [[(r, c) for r, c in path] for path in solution] if solution is not None else None
        )

        template: LevelTemplate = build_level_template(
            grid=self._source_grid,
            level_id=self._source_level_id,
            variant=self._variant_override,
            bridges=self._source_bridges,
            generator=generator_normalized,
            reward_config=self._reward_config_override,
            render_config=self._render_config_override,
            palette=self._palette_override,
            solution=self._source_solution,
        )

        single_observation_space: spaces.Box
        single_action_space: spaces.Discrete
        single_observation_space, single_action_space = self._load_template(template)

        super().__init__()
        self.num_envs = self._num_envs
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, self._num_envs)
        self.action_space = batch_space(single_action_space, self._num_envs)

        self._reset_masked(np.ones((self._num_envs,), dtype=np.bool_))

    @staticmethod
    def _normalize_generator_config(generator: GeneratorConfig | None) -> GeneratorConfig | None:
        """Ensure generator configurations using bridges fall back to random walk mode."""
        if generator is None:
            return None
        if generator.bridges_probability and generator.mode != "random_walk":
            return replace(generator, mode="random_walk")
        return generator

    def _load_template(self, template: LevelTemplate) -> tuple[spaces.Box, spaces.Discrete]:
        """Load a :class:`numberlink.level_setup.LevelTemplate` and initialize derived state."""
        self._template: LevelTemplate = template
        self.variant: VariantConfig = template.variant
        self._reward_cfg: RewardConfig = template.reward_config
        self._render_cfg: RenderConfig = template.render_config
        self.level_id: str | None = template.level_id

        self._height: int = template.height
        self._width: int = template.width
        self.H: int = self._height
        self.W: int = self._width
        self._num_colors: int = template.num_colors
        self.num_colors: int = self._num_colors
        self._num_dirs: int = template.num_dirs
        self._actions_per_color: int = template.actions_per_color
        self._action_size: int = template.action_space_size
        self._cell_action_size: int = template.cell_switch_action_space_size
        self._cell_action_stride: int = self._num_colors + 1

        max_dim: int = max(self._height, self._width)
        self._coord_dtype: type[np.signedinteger] = select_signed_dtype(-1, max_dim)
        self._stack_capacity: int = self._height * self._width
        self._stack_len_dtype: type[np.signedinteger] = select_signed_dtype(-2, self._stack_capacity)
        self._color_code_dtype: type[np.unsignedinteger] = select_unsigned_dtype(self._num_colors)
        self._color_index_dtype: type[np.unsignedinteger] = select_unsigned_dtype(max(self._num_colors - 1, 0))
        self._dir_dtype: type[np.unsignedinteger] = select_unsigned_dtype(max(self._num_dirs - 1, 0))
        self._head_dtype: type[np.unsignedinteger] = select_unsigned_dtype(1)
        self._action_index_dtype: type[np.unsignedinteger] = select_unsigned_dtype(max(self._action_size - 1, 0))
        self._cell_index_dtype: type[np.unsignedinteger] = select_unsigned_dtype(max(self._cell_action_size - 1, 0))

        self._dirs: NDArray[np.signedinteger] = template.dirs.astype(self._coord_dtype, copy=False)
        self._dir_to_index: dict[Coord, int] = {
            (int(vec[0]), int(vec[1])): int(idx) for idx, vec in enumerate(self._dirs)
        }
        self._endpoints: NDArray[np.signedinteger] = template.endpoints.astype(self._coord_dtype, copy=False)
        self._endpoint_mask: NDArray[np.bool_] = np.zeros((self._height, self._width), dtype=np.bool_)
        ep_rows: NDArray[np.intp] = self._endpoints[:, :, 0].reshape(-1).astype(np.intp, copy=False)
        ep_cols: NDArray[np.intp] = self._endpoints[:, :, 1].reshape(-1).astype(np.intp, copy=False)
        self._endpoint_mask[ep_rows, ep_cols] = True
        self._bridges_mask: NDArray[np.bool_] = template.bridges_mask.copy()

        self.max_steps: int = (
            self._step_limit_override if self._step_limit_override is not None else 10 * self._stack_capacity
        )
        self._step_count_dtype: type[np.unsignedinteger] = select_unsigned_dtype(self.max_steps)

        pixels_per_cell_h_candidate: int = 1
        pixels_per_cell_w_candidate: int = 1
        if self._render_cfg.render_height is not None:
            if self._render_cfg.render_height < self._height:
                raise ValueError(
                    f"render_height ({self._render_cfg.render_height}) must be >= grid height ({self._height})"
                )
            pixels_per_cell_h_candidate = max(1, self._render_cfg.render_height // self._height)
        if self._render_cfg.render_width is not None:
            if self._render_cfg.render_width < self._width:
                raise ValueError(
                    f"render_width ({self._render_cfg.render_width}) must be >= grid width ({self._width})"
                )
            pixels_per_cell_w_candidate = max(1, self._render_cfg.render_width // self._width)
        pixels_per_cell: int = min(pixels_per_cell_h_candidate, pixels_per_cell_w_candidate)
        self._pixels_per_cell_h: int = pixels_per_cell
        self._pixels_per_cell_w: int = pixels_per_cell
        obs_height: int = self._pixels_per_cell_h * self._height
        obs_width: int = self._pixels_per_cell_w * self._width

        single_observation_space: spaces.Box = spaces.Box(
            low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )
        single_action_space: spaces.Discrete = spaces.Discrete(
            self._cell_action_size if self.variant.cell_switching_mode else self._action_size
        )

        self._palette: NDArray[np.uint8] = np.stack(template.palette_arrays, axis=0)
        self._letters: list[str] = list(template.letters)
        self._stack_index: NDArray[np.signedinteger] = np.arange(self._stack_capacity, dtype=self._stack_len_dtype)

        self._base_grid: NDArray[np.unsignedinteger] = np.zeros(
            (self._height, self._width), dtype=self._color_code_dtype
        )
        self._base_lane_v: NDArray[np.unsignedinteger] = np.zeros(
            (self._height, self._width), dtype=self._color_code_dtype
        )
        self._base_lane_h: NDArray[np.unsignedinteger] = np.zeros(
            (self._height, self._width), dtype=self._color_code_dtype
        )
        self._stack_rows_base: NDArray[np.signedinteger] = np.zeros(
            (self._num_colors, 2, self._stack_capacity), dtype=self._coord_dtype
        )
        self._stack_cols_base: NDArray[np.signedinteger] = np.zeros_like(self._stack_rows_base)
        self._stack_lane_base: NDArray[np.uint8] = np.zeros((self._num_colors, 2, self._stack_capacity), dtype=np.uint8)
        self._stack_len_base: NDArray[np.signedinteger] = np.ones((self._num_colors, 2), dtype=self._stack_len_dtype)
        self._heads_base: NDArray[np.signedinteger] = np.zeros((self._num_colors, 2, 2), dtype=self._coord_dtype)
        self._arm_presence_base: NDArray[np.bool_] = np.zeros(
            (self._num_colors, 2, self._height, self._width), dtype=np.bool_
        )

        for color_idx in range(self._num_colors):
            color_code: int = color_idx + 1
            for head_idx in (0, 1):
                row: int = int(self._endpoints[color_idx, head_idx, 0])
                col: int = int(self._endpoints[color_idx, head_idx, 1])
                lane_code: np.uint8 = LANE_NORMAL
                if self._bridges_mask[row, col]:
                    self._base_lane_v[row, col] = color_code
                    self._base_lane_h[row, col] = color_code
                    lane_code = LANE_BOTH
                else:
                    self._base_grid[row, col] = color_code
                self._stack_rows_base[color_idx, head_idx, 0] = row
                self._stack_cols_base[color_idx, head_idx, 0] = col
                self._stack_lane_base[color_idx, head_idx, 0] = lane_code
                self._heads_base[color_idx, head_idx, 0] = row
                self._heads_base[color_idx, head_idx, 1] = col
                self._arm_presence_base[color_idx, head_idx, row, col] = True

        action_indices: NDArray[np.unsignedinteger] = np.arange(
            max(self._action_size, 1), dtype=self._action_index_dtype
        )
        self._decode_color: NDArray[np.unsignedinteger] = np.zeros_like(action_indices, dtype=self._color_index_dtype)
        self._decode_head: NDArray[np.unsignedinteger] = np.zeros_like(action_indices, dtype=self._head_dtype)
        self._decode_dir: NDArray[np.unsignedinteger] = np.zeros_like(action_indices, dtype=self._dir_dtype)
        if self._action_size > 0:
            self._decode_color = (action_indices // self._actions_per_color).astype(self._color_index_dtype, copy=False)
            remainder: NDArray[np.unsignedinteger] = action_indices % self._actions_per_color
            self._decode_head = (remainder // self._num_dirs).astype(self._head_dtype, copy=False)
            self._decode_dir = (remainder % self._num_dirs).astype(self._dir_dtype, copy=False)

        cell_indices: NDArray[np.unsignedinteger] = np.arange(
            max(self._cell_action_size, 1), dtype=self._cell_index_dtype
        )
        self._decode_cell_row: NDArray[np.signedinteger] = np.zeros_like(cell_indices, dtype=self._coord_dtype)
        self._decode_cell_col: NDArray[np.signedinteger] = np.zeros_like(cell_indices, dtype=self._coord_dtype)
        self._decode_cell_color: NDArray[np.unsignedinteger] = np.zeros_like(cell_indices, dtype=self._color_code_dtype)
        if self._cell_action_size > 0:
            colors_plus_clear: int = self._cell_action_stride
            flat_cells: NDArray[np.unsignedinteger] = cell_indices // colors_plus_clear
            self._decode_cell_row = (flat_cells // self._width).astype(self._coord_dtype, copy=False)
            self._decode_cell_col = (flat_cells % self._width).astype(self._coord_dtype, copy=False)
            self._decode_cell_color = (cell_indices % colors_plus_clear).astype(self._color_code_dtype, copy=False)

        self._grid_codes: NDArray[np.unsignedinteger] = np.zeros(
            (self._num_envs, self._height, self._width), dtype=self._color_code_dtype
        )
        self._lane_v: NDArray[np.unsignedinteger] = np.zeros(
            (self._num_envs, self._height, self._width), dtype=self._color_code_dtype
        )
        self._lane_h: NDArray[np.unsignedinteger] = np.zeros(
            (self._num_envs, self._height, self._width), dtype=self._color_code_dtype
        )
        self._stack_rows: NDArray[np.signedinteger] = np.zeros(
            (self._num_envs, self._num_colors, 2, self._stack_capacity), dtype=self._coord_dtype
        )
        self._stack_cols: NDArray[np.signedinteger] = np.zeros_like(self._stack_rows)
        self._stack_lane: NDArray[np.uint8] = np.zeros(
            (self._num_envs, self._num_colors, 2, self._stack_capacity), dtype=np.uint8
        )
        self._stack_len: NDArray[np.signedinteger] = np.zeros(
            (self._num_envs, self._num_colors, 2), dtype=self._stack_len_dtype
        )
        self._heads: NDArray[np.signedinteger] = np.zeros(
            (self._num_envs, self._num_colors, 2, 2), dtype=self._coord_dtype
        )
        self._closed: NDArray[np.bool_] = np.zeros((self._num_envs, self._num_colors), dtype=np.bool_)
        self._step_count: NDArray[np.unsignedinteger] = np.zeros((self._num_envs,), dtype=self._step_count_dtype)
        self._done_mask: NDArray[np.bool_] = np.zeros((self._num_envs,), dtype=np.bool_)
        self._arm_presence: NDArray[np.bool_] = np.zeros(
            (self._num_envs, self._num_colors, 2, self._height, self._width), dtype=np.bool_
        )
        self._env_indices: NDArray[np.intp] = np.arange(self._num_envs, dtype=np.intp)
        self._tmp_color_indices: NDArray[np.unsignedinteger] = np.zeros(
            (self._num_envs,), dtype=self._color_index_dtype
        )
        self._tmp_head_indices: NDArray[np.unsignedinteger] = np.zeros((self._num_envs,), dtype=self._head_dtype)
        self._tmp_dir_indices: NDArray[np.unsignedinteger] = np.zeros((self._num_envs,), dtype=self._dir_dtype)
        self._path_action_mask: NDArray[np.uint8] = np.zeros((self._num_envs, self._action_size), dtype=np.uint8)
        self._allow_buffer: NDArray[np.bool_] = np.zeros((self._num_envs,), dtype=np.bool_)
        self._apply_buffer: NDArray[np.bool_] = np.zeros((self._num_envs,), dtype=np.bool_)
        self._visit_buffer: NDArray[np.bool_] = np.zeros((self._height, self._width), dtype=np.bool_)
        self._visit_buffer_alt: NDArray[np.bool_] = np.zeros((self._height, self._width), dtype=np.bool_)
        self._bfs_queue: deque[Coord] = deque()

        self._static_cell_mask: NDArray[np.uint8] | None = (
            self._build_cell_switch_mask() if self.variant.cell_switching_mode else None
        )

        self._solution_coords: list[list[Coord]] | None = (
            [[(int(r), int(c)) for r, c in path] for path in template.solution]
            if template.solution is not None
            else None
        )
        self._solution_actions: list[ActType] | None = self._compute_solution_actions()

        return single_observation_space, single_action_space

    def _compute_solution_actions(self) -> list[ActType] | None:
        """Return solution actions derived from precomputed coordinate paths when available."""
        if self._solution_coords is None:
            return None

        actions: list[ActType]
        if self.variant.cell_switching_mode:
            actions = []
            for color_idx, path in enumerate(self._solution_coords):
                if len(path) <= 2:
                    continue
                for row, col in path[1:-1]:
                    try:
                        actions.append(self.encode_cell_switching_action(row, col, color_idx + 1))
                    except ValueError:
                        continue
            return actions if actions else None

        actions = []
        for color_idx, path in enumerate(self._solution_coords):
            if len(path) <= 1:
                continue
            ep0: Coord = (int(self._endpoints[color_idx, 0, 0]), int(self._endpoints[color_idx, 0, 1]))
            ep1: Coord = (int(self._endpoints[color_idx, 1, 0]), int(self._endpoints[color_idx, 1, 1]))
            if path[0] == ep0:
                head_idx = 0
            elif path[0] == ep1:
                head_idx = 1
            else:
                continue
            for idx in range(len(path) - 1):
                cur_r, cur_c = path[idx]
                nxt_r, nxt_c = path[idx + 1]
                dr: int = int(nxt_r) - int(cur_r)
                dc: int = int(nxt_c) - int(cur_c)
                dir_idx: int | None = self._dir_to_index.get((dr, dc))
                if dir_idx is None:
                    continue
                action_idx: int = color_idx * self._actions_per_color + head_idx * self._num_dirs + dir_idx
                actions.append(action_idx)

        return actions if actions else None

    def encode_cell_switching_action(self, row: int, col: int, color_value: int) -> int:
        """Encode a cell assignment into the flat action index used in cell switching mode."""
        colors_plus_clear: int = self._num_colors + 1
        if not (0 <= row < self._height and 0 <= col < self._width and 0 <= color_value < colors_plus_clear):
            raise ValueError("Cell switching action components out of range")
        return (row * self._width + col) * colors_plus_clear + color_value

    def get_solution(self) -> list[ActType] | None:
        """Return a copy of the precomputed solution action list when available."""
        if self._solution_actions is None:
            return None
        return list(self._solution_actions)

    def regenerate_level(self, seed: int | None = None) -> tuple[ObsType, InfoDict]:
        """Regenerate the current template using the stored generator configuration."""
        if self._generator_config is None:
            raise RuntimeError("regenerate_level requires an environment created with a generator configuration")

        base_generator: GeneratorConfig = self._generator_config
        generator_seed: int | None = (
            seed if seed is not None else (None if base_generator.seed is None else base_generator.seed + 1)
        )
        new_generator: GeneratorConfig = replace(base_generator, seed=generator_seed)
        normalized_generator: GeneratorConfig | None = self._normalize_generator_config(new_generator)

        template: LevelTemplate = build_level_template(
            grid=self._source_grid,
            level_id=self._source_level_id,
            variant=self._variant_override,
            bridges=self._source_bridges,
            generator=normalized_generator,
            reward_config=self._reward_config_override,
            render_config=self._render_config_override,
            palette=self._palette_override,
            solution=self._source_solution,
        )

        single_observation_space: spaces.Box
        single_action_space: spaces.Discrete
        single_observation_space, single_action_space = self._load_template(template)
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, self._num_envs)
        self.action_space = batch_space(single_action_space, self._num_envs)

        self._generator_config = normalized_generator
        return self.reset(seed=seed)

    # Gymnasium API

    def reset(
        self, *, seed: int | None = None, options: dict[str, NDArray[np.bool_]] | None = None
    ) -> tuple[ObsType, InfoDict]:
        """Reset selected environments and return initial observations and info.

        When ``options`` contains a ``'reset_mask'`` key it must be a boolean array with shape ``(num_envs,)``. Only
        environments with a ``True`` entry are reset. The reset operation is performed by :meth:`_reset_masked`. The
        returned info dictionary is produced by :meth:`_build_info`.

        :param seed: Random seed for reproducibility across all environments.
        :param options: Optional configuration dictionary. If provided it may contain ``'reset_mask'`` with a boolean
            array of shape ``(num_envs,)``.
        :return: Tuple ``(observations, infos)`` where ``observations`` is an array of shape
            ``(num_envs, height, width, 3)`` and ``infos`` is the dictionary returned by :meth:`_build_info`.
        :rtype: tuple[ObsType, InfoDict]
        :raises ValueError: If ``reset_mask`` does not have shape ``(num_envs,)``.
        """
        if seed is not None:
            super().reset(seed=seed)

        mask: NDArray[np.bool_] = (
            options["reset_mask"] if options and "reset_mask" in options else np.ones((self._num_envs,), dtype=np.bool_)
        )

        if mask.shape != (self._num_envs,):
            raise ValueError(f"reset_mask must have shape ({self._num_envs},)")

        self._reset_masked(mask)
        return self._render_rgb(), self._build_info(action_mask=self._compute_action_mask())

    def step(
        self, actions: NDArray[np.integer]
    ) -> tuple[ObsType, NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], InfoDict]:
        """Advance all environments by one step and return batched outputs.

        In path mode actions encode ``(color, head, direction)`` and are decoded using lookup arrays prepared at
        initialization. In cell switching mode actions encode ``(row, col, color)`` and are handled by
        :meth:`_step_cell_switch`.

        After applying actions the method updates step counters, computes rewards, determines terminal and truncation
        masks, and constructs the info dictionary using :meth:`_build_info`.

        :param actions: Array of action indices for each environment with shape ``(num_envs,)``.
        :return: Tuple ``(observations, rewards, terminations, truncations, infos)``.
        :rtype: tuple[ObsType, NDArray[numpy.float32], NDArray[bool], NDArray[bool], InfoDict]
        """
        act: NDArray[np.integer] = actions.reshape(self._num_envs)

        if np.any(self._done_mask):
            self._reset_masked(self._done_mask.copy())

        connected_before: NDArray[np.int_] = self._closed.sum(axis=1)

        valid: NDArray[np.bool_] = (
            self._step_cell_switch(act) if self.variant.cell_switching_mode else self._step_path(act)
        )

        self._step_count += 1

        connected_after: NDArray[np.int_] = self._closed.sum(axis=1)
        delta: NDArray[np.int_] = np.maximum(connected_after - connected_before, 0)

        rewards: NDArray[np.float32] = np.full((self._num_envs,), self._reward_cfg.step_penalty, dtype=np.float32)
        rewards[~valid] += self._reward_cfg.invalid_penalty
        if np.any(delta):
            rewards += delta * self._reward_cfg.connect_bonus

        solved: NDArray[np.bool_] = self._compute_solved_mask()
        rewards[solved] += self._reward_cfg.win_bonus

        action_mask: NDArray[np.uint8] = self._compute_action_mask()
        deadlocked: NDArray[np.bool_] = (~solved) & (action_mask.sum(axis=1) == 0)
        rewards[deadlocked] += 2.0 * self._reward_cfg.invalid_penalty

        terminated: NDArray[np.bool_] = solved | deadlocked
        truncated: NDArray[np.bool_] = self._step_count >= self.max_steps

        self._done_mask = terminated | truncated

        return (
            self._render_rgb(),
            rewards,
            terminated,
            truncated,
            self._build_info(action_mask=action_mask, solved=solved, deadlocked=deadlocked),
        )

    def render(self) -> tuple[RenderFrame, ...] | None:  # pyright: ignore[reportInvalidTypeVarUse]
        """Return rendered RGB frames for all environments as a tuple.

        Each frame is an array of shape ``(height, width, 3)`` with type ``uint8``. The visual rules for endpoints and
        bridges are implemented in :meth:`_render_rgb`.

        :return: Tuple of RGB frames, one per environment, or ``None`` when rendering is not available for the
            configured mode.
        :rtype: tuple[numberlink.types.RenderFrame, ...] | None
        """
        return tuple(self._render_rgb())

    def close(self, **kwargs: dict[str, Any]) -> None:
        """Perform environment cleanup for API compatibility.

        The vectorized environment stores only in-memory NumPy arrays and does not require explicit cleanup. This method
        exists for API compatibility with :class:`gymnasium.Env`.

        :param kwargs: Additional keyword arguments accepted for API compatibility.
        :type kwargs: dict[str, Any]
        """
        pass

    # Reset helpers

    def _reset_masked(self, mask: NDArray[np.bool_]) -> None:
        """Reset internal state for a subset of environments indicated by mask.

        The method restores the selected environments to the state derived from the
        :class:`numberlink.level_setup.LevelTemplate` that was prepared during initialization.

        :param mask: Boolean array of shape ``(num_envs,)`` indicating which environments to reset.
        :type mask: NDArray[bool]
        """
        if not np.any(mask):
            return

        idx: NDArray[np.intp] = np.where(mask)[0]
        self._grid_codes[idx] = self._base_grid
        self._lane_v[idx] = self._base_lane_v
        self._lane_h[idx] = self._base_lane_h
        self._stack_rows[idx] = self._stack_rows_base
        self._stack_cols[idx] = self._stack_cols_base
        self._stack_lane[idx] = self._stack_lane_base
        self._stack_len[idx] = self._stack_len_base
        self._heads[idx] = self._heads_base
        self._closed[idx] = False
        self._step_count[idx] = 0
        self._done_mask[idx] = False
        self._arm_presence[idx] = self._arm_presence_base

    # Step helpers

    def _step_cell_switch(self, actions: NDArray[np.integer]) -> NDArray[np.bool_]:
        """Apply cell switching actions by setting cell colors directly.

        Actions are decoded to ``(row, col, color)`` triples using lookup arrays created at initialization. Endpoint
        cells are not modified. The method writes into the grid or lane arrays and returns a bool mask of valid actions.

        :param actions: Array of action indices for each environment.
        :type actions: NDArray[np.integer]
        :return: Boolean array indicating which actions were valid and applied.
        :rtype: NDArray[bool]
        """
        result: NDArray[np.bool_] = np.zeros((self._num_envs,), dtype=np.bool_)
        if self._cell_action_size == 0:
            return result

        in_range: NDArray[np.bool_] = (actions >= 0) & (actions < self._cell_action_size)
        if not np.any(in_range):
            return result

        sel: NDArray[np.intp] = np.where(in_range)[0]
        rows: NDArray[np.intp] = self._decode_cell_row[actions[sel]].astype(np.intp, copy=False)
        cols: NDArray[np.intp] = self._decode_cell_col[actions[sel]].astype(np.intp, copy=False)
        values: NDArray[np.unsignedinteger] = self._decode_cell_color[actions[sel]].astype(
            self._color_code_dtype, copy=False
        )

        endpoint_blocked: NDArray[np.bool_] = self._endpoint_mask[rows, cols]
        if np.all(endpoint_blocked):
            return result

        modifiable_idx: NDArray[np.intp] = np.where(~endpoint_blocked)[0]
        env_sel: NDArray[np.intp] = sel[modifiable_idx]
        row_sel: NDArray[np.intp] = rows[modifiable_idx]
        col_sel: NDArray[np.intp] = cols[modifiable_idx]
        val_sel: NDArray[np.unsignedinteger] = values[modifiable_idx]

        affected_colors: set[int] = set()
        for idx, env in enumerate(env_sel):
            r = int(row_sel[idx])
            c = int(col_sel[idx])
            new_code: int = int(val_sel[idx])

            if not self._bridges_mask[r, c]:
                prev_code: int = int(self._grid_codes[env, r, c])
                if prev_code > 0:
                    affected_colors.add(prev_code - 1)
                self._grid_codes[env, r, c] = new_code
            else:
                prev_v: int = int(self._lane_v[env, r, c])
                prev_h: int = int(self._lane_h[env, r, c])
                if prev_v > 0:
                    affected_colors.add(prev_v - 1)
                if prev_h > 0:
                    affected_colors.add(prev_h - 1)
                self._lane_v[env, r, c] = new_code
                self._lane_h[env, r, c] = new_code

            if new_code > 0:
                affected_colors.add(new_code - 1)
            result[env] = True

        if affected_colors:
            self._recompute_closed(np.array(sorted(affected_colors), dtype=np.int32))

        return result

    def _step_path(self, actions: NDArray[np.integer]) -> NDArray[np.bool_]:
        """Apply path mode actions to extend or retract color paths.

        The method decodes actions to ``(color, head, direction)`` using lookup arrays and validates moves using
        :meth:`_can_occupy_targets`. It handles backtracking by calling :meth:`_perform_backtrack`, and handles joins
        and endpoint connections by pushing to stacks with :meth:`_push_stack` and updating occupancy with
        :meth:`_occupy_targets`.

        :param actions: Array of action indices for each environment.
        :type actions: NDArray[np.integer]
        :return: Boolean array indicating which actions were valid and applied.
        :rtype: NDArray[bool]
        """
        valid: NDArray[np.bool_] = np.zeros((self._num_envs,), dtype=np.bool_)
        if self._action_size == 0:
            return valid

        env_idx: NDArray[np.intp] = self._env_indices

        in_range: NDArray[np.bool_] = (actions >= 0) & (actions < self._action_size)

        colors: NDArray[np.unsignedinteger] = self._tmp_color_indices
        colors.fill(0)
        heads: NDArray[np.unsignedinteger] = self._tmp_head_indices
        heads.fill(0)
        dirs: NDArray[np.unsignedinteger] = self._tmp_dir_indices
        dirs.fill(0)
        colors[in_range] = self._decode_color[actions[in_range]]
        heads[in_range] = self._decode_head[actions[in_range]]
        dirs[in_range] = self._decode_dir[actions[in_range]]

        current_heads: NDArray[np.signedinteger] = self._heads[env_idx, colors, heads]
        deltas: NDArray[np.signedinteger] = self._dirs[dirs]
        targets: NDArray[np.signedinteger] = current_heads + deltas
        target_rows: NDArray[np.signedinteger] = targets[:, 0]
        target_cols: NDArray[np.signedinteger] = targets[:, 1]

        in_bounds: NDArray[np.bool_] = (
            (target_rows >= 0) & (target_rows < self._height) & (target_cols >= 0) & (target_cols < self._width)
        )
        active: NDArray[np.bool_] = in_range & in_bounds

        stack_len: NDArray[np.signedinteger] = self._stack_len[env_idx, colors, heads]
        has_prev: NDArray[np.bool_] = stack_len >= 2
        prev_idx: NDArray[np.signedinteger] = np.clip(stack_len - 2, 0, self._stack_capacity - 1)
        prev_rows: NDArray[np.signedinteger] = self._stack_rows[env_idx, colors, heads, prev_idx]
        prev_cols: NDArray[np.signedinteger] = self._stack_cols[env_idx, colors, heads, prev_idx]
        backtrack: NDArray[np.bool_] = active & has_prev & (target_rows == prev_rows) & (target_cols == prev_cols)

        if np.any(backtrack):
            bt_idx: NDArray[np.intp] = np.where(backtrack)[0]
            self._perform_backtrack(env_idx[bt_idx], colors[bt_idx], heads[bt_idx])
            valid[bt_idx] = True

        remaining: NDArray[np.bool_] = active & ~backtrack

        other_heads: NDArray[np.signedinteger] = self._heads[env_idx, colors, 1 - heads]
        join_head: NDArray[np.bool_] = (
            remaining & (target_rows == other_heads[:, 0]) & (target_cols == other_heads[:, 1])
        )
        color_codes_full: NDArray[np.unsignedinteger] = colors.astype(self._color_code_dtype, copy=False) + np.array(
            1, dtype=self._color_code_dtype
        )

        lane_codes: NDArray[np.uint8]
        if np.any(join_head):
            jh_idx: NDArray[np.intp] = np.where(join_head)[0]
            lane_codes = self._lane_codes(dirs[jh_idx], target_rows[jh_idx], target_cols[jh_idx])
            self._occupy_targets(
                env_idx[jh_idx], target_rows[jh_idx], target_cols[jh_idx], color_codes_full[jh_idx], lane_codes
            )
            self._push_stack(
                env_idx[jh_idx], colors[jh_idx], heads[jh_idx], target_rows[jh_idx], target_cols[jh_idx], lane_codes
            )
            valid[jh_idx] = True
        remaining &= ~join_head

        other_endpoints: NDArray[np.signedinteger] = self._endpoints[colors, 1 - heads]
        other_stack_len: NDArray[np.signedinteger] = self._stack_len[env_idx, colors, 1 - heads]
        join_endpoint: NDArray[np.bool_] = (
            remaining
            & (target_rows == other_endpoints[:, 0])
            & (target_cols == other_endpoints[:, 1])
            & (other_stack_len == 1)
        )
        if np.any(join_endpoint):
            je_idx: NDArray[np.intp] = np.where(join_endpoint)[0]
            lane_codes = self._lane_codes(dirs[je_idx], target_rows[je_idx], target_cols[je_idx])
            self._occupy_targets(
                env_idx[je_idx], target_rows[je_idx], target_cols[je_idx], color_codes_full[je_idx], lane_codes
            )
            self._push_stack(
                env_idx[je_idx], colors[je_idx], heads[je_idx], target_rows[je_idx], target_cols[je_idx], lane_codes
            )
            valid[je_idx] = True

        remaining &= ~join_endpoint

        if np.any(remaining):
            rem_idx: NDArray[np.intp] = np.where(remaining)[0]
            rows: NDArray[np.signedinteger] = target_rows[rem_idx]
            cols: NDArray[np.signedinteger] = target_cols[rem_idx]
            color_codes: NDArray[np.unsignedinteger] = color_codes_full[rem_idx]
            dir_idx: NDArray[np.unsignedinteger] = dirs[rem_idx]

            can_occupy: NDArray[np.bool_] = self._can_occupy_targets(env_idx[rem_idx], rows, cols, color_codes, dir_idx)
            if np.any(can_occupy):
                occ_idx: NDArray[np.intp] = rem_idx[np.where(can_occupy)[0]]
                occ_rows: NDArray[np.intp] = target_rows[occ_idx]
                occ_cols: NDArray[np.intp] = target_cols[occ_idx]
                occ_dirs: NDArray[np.unsignedinteger] = dirs[occ_idx]
                occ_lanes: NDArray[np.uint8] = self._lane_codes(occ_dirs, occ_rows, occ_cols)
                interior: NDArray[np.bool_] = self._occupies_other_stack(
                    env_idx[occ_idx], colors[occ_idx], heads[occ_idx], occ_rows, occ_cols
                )
                apply_mask: NDArray[np.bool_] = ~interior
                if np.any(apply_mask):
                    sel: NDArray[np.intp] = occ_idx[apply_mask]
                    self._occupy_targets(
                        env_idx[sel], target_rows[sel], target_cols[sel], color_codes_full[sel], occ_lanes[apply_mask]
                    )
                    self._push_stack(
                        env_idx[sel], colors[sel], heads[sel], target_rows[sel], target_cols[sel], occ_lanes[apply_mask]
                    )
                    valid[sel] = True

        self._update_heads()
        affected_colors: NDArray[np.integer] = np.unique(colors[valid]).astype(np.int32, copy=False)
        if affected_colors.size > 0:
            self._recompute_closed(affected_colors)

        return valid

    # Stack / occupancy helpers

    def _perform_backtrack(
        self, env_idx: NDArray[np.intp], color_idx: NDArray[np.integer], head_idx: NDArray[np.integer]
    ) -> None:
        """Remove the most recent path step for the specified heads.

        The method removes the most recent entry from the stack for each ``(env_idx, color_idx, head_idx)`` triple
        unless the entry is the original endpoint. Occupancy is cleared from the grid or lane arrays depending on
        whether the position is a bridge cell.

        :param env_idx: Environment indices to modify.
        :param color_idx: Color indices for the paths to backtrack.
        :param head_idx: Head indices indicating which endpoint to backtrack.
        """
        if env_idx.size == 0:
            return

        lengths: NDArray[np.signedinteger] = self._stack_len[env_idx, color_idx, head_idx]
        top_idx: NDArray[np.signedinteger] = np.clip(lengths - 1, 0, self._stack_capacity - 1)
        rows: NDArray[np.signedinteger] = self._stack_rows[env_idx, color_idx, head_idx, top_idx]
        cols: NDArray[np.signedinteger] = self._stack_cols[env_idx, color_idx, head_idx, top_idx]
        lanes: NDArray[np.uint8] = self._stack_lane[env_idx, color_idx, head_idx, top_idx]
        endpoints: NDArray[np.signedinteger] = self._endpoints[color_idx]
        is_endpoint: NDArray[np.bool_] = ((rows == endpoints[:, 0, 0]) & (cols == endpoints[:, 0, 1])) | (
            (rows == endpoints[:, 1, 0]) & (cols == endpoints[:, 1, 1])
        )
        removable: NDArray[np.bool_] = ~is_endpoint
        if not np.any(removable):
            return

        env_sel: NDArray[np.intp] = env_idx[removable]
        color_sel: NDArray[np.integer] = color_idx[removable]
        head_sel: NDArray[np.integer] = head_idx[removable]
        row_sel: NDArray[np.integer] = rows[removable]
        col_sel: NDArray[np.integer] = cols[removable]
        lane_sel: NDArray[np.uint8] = lanes[removable]

        bridge_mask: NDArray[np.bool_] = self._bridges_mask[row_sel, col_sel]
        normal_mask: NDArray[np.bool_] = ~bridge_mask
        if np.any(normal_mask):
            nm: NDArray[np.intp] = np.where(normal_mask)[0]
            self._grid_codes[env_sel[nm], row_sel[nm], col_sel[nm]] = 0
        if np.any(bridge_mask):
            bm: NDArray[np.intp] = np.where(bridge_mask)[0]
            vertical: NDArray[np.bool_] = lane_sel[bm] == LANE_VERTICAL
            horizontal: NDArray[np.bool_] = lane_sel[bm] == LANE_HORIZONTAL
            both: NDArray[np.bool_] = lane_sel[bm] == LANE_BOTH
            if np.any(vertical):
                vi: NDArray[np.intp] = bm[vertical]
                self._lane_v[env_sel[vi], row_sel[vi], col_sel[vi]] = 0
            if np.any(horizontal):
                hi: NDArray[np.intp] = bm[horizontal]
                self._lane_h[env_sel[hi], row_sel[hi], col_sel[hi]] = 0
            if np.any(both):
                bi: NDArray[np.intp] = bm[both]
                self._lane_v[env_sel[bi], row_sel[bi], col_sel[bi]] = 0
                self._lane_h[env_sel[bi], row_sel[bi], col_sel[bi]] = 0

        self._arm_presence[env_sel, color_sel, head_sel, row_sel, col_sel] = False
        self._stack_len[env_sel, color_sel, head_sel] -= 1

    def _occupy_targets(
        self,
        env_idx: NDArray[np.intp],
        rows: NDArray[np.integer],
        cols: NDArray[np.integer],
        color_codes: NDArray[np.unsignedinteger],
        lane_codes: NDArray[np.uint8],
    ) -> None:
        """Set occupancy for the provided target positions and lane types.

        For regular cells the grid code array is written. For bridge cells the appropriate lane arrays are updated
        according to ``lane_codes``.

        :param env_idx: Environment indices to modify.
        :param rows: Row coordinates of cells to occupy.
        :param cols: Column coordinates of cells to occupy.
        :param color_codes: Color code values to place in the cells.
        :param lane_codes: Lane type codes produced by :meth:`_lane_codes`.
        """
        if env_idx.size == 0:
            return

        bridge_mask: NDArray[np.bool_] = self._bridges_mask[rows, cols]
        normal_mask: NDArray[np.bool_] = ~bridge_mask
        if np.any(normal_mask):
            nm: NDArray[np.intp] = np.where(normal_mask)[0]
            self._grid_codes[env_idx[nm], rows[nm], cols[nm]] = color_codes[nm]
        if np.any(bridge_mask):
            bm: NDArray[np.intp] = np.where(bridge_mask)[0]
            vertical: NDArray[np.bool_] = lane_codes[bm] == LANE_VERTICAL
            horizontal: NDArray[np.bool_] = lane_codes[bm] == LANE_HORIZONTAL
            both: NDArray[np.bool_] = lane_codes[bm] == LANE_BOTH
            if np.any(vertical):
                vi: NDArray[np.intp] = bm[vertical]
                self._lane_v[env_idx[vi], rows[vi], cols[vi]] = color_codes[vi]
            if np.any(horizontal):
                hi: NDArray[np.intp] = bm[horizontal]
                self._lane_h[env_idx[hi], rows[hi], cols[hi]] = color_codes[hi]
            if np.any(both):
                bi: NDArray[np.intp] = bm[both]
                self._lane_v[env_idx[bi], rows[bi], cols[bi]] = color_codes[bi]
                self._lane_h[env_idx[bi], rows[bi], cols[bi]] = color_codes[bi]

    def _push_stack(
        self,
        env_idx: NDArray[np.intp],
        color_idx: NDArray[np.integer],
        head_idx: NDArray[np.integer],
        rows: NDArray[np.signedinteger],
        cols: NDArray[np.signedinteger],
        lane_codes: NDArray[np.uint8],
    ) -> None:
        """Push positions onto the per-color per-head stacks.

        The stack arrays record path history for each head. This method stores the provided positions at the current
        stack length indices and increments the stack lengths accordingly.

        :param env_idx: Environment indices to modify.
        :param color_idx: Color indices for the paths being extended.
        :param head_idx: Head indices (0 or 1) specifying which endpoint is moving.
        :param rows: Row coordinates to push onto stacks.
        :param cols: Column coordinates to push onto stacks.
        :param lane_codes: Lane type codes associated with the positions.
        """
        if env_idx.size == 0:
            return

        positions: NDArray[np.integer] = self._stack_len[env_idx, color_idx, head_idx]
        self._stack_rows[env_idx, color_idx, head_idx, positions] = rows
        self._stack_cols[env_idx, color_idx, head_idx, positions] = cols
        self._stack_lane[env_idx, color_idx, head_idx, positions] = lane_codes
        self._stack_len[env_idx, color_idx, head_idx] = positions + 1
        self._arm_presence[env_idx, color_idx, head_idx, rows, cols] = True

    def _can_occupy_targets(
        self,
        env_idx: NDArray[np.intp],
        rows: NDArray[np.integer],
        cols: NDArray[np.integer],
        color_codes: NDArray[np.unsignedinteger],
        dir_idx: NDArray[np.unsignedinteger],
    ) -> NDArray[np.bool_]:
        """Return a boolean mask indicating which targets can be occupied.

        For regular cells a target is occupiable when the grid code is zero. For bridge cells occupancy is allowed when
        the relevant lane is empty or already contains the same color code.

        :param env_idx: Environment indices to check.
        :param rows: Row coordinates of target cells.
        :param cols: Column coordinates of target cells.
        :param color_codes: Color codes attempting to occupy cells.
        :param dir_idx: Direction indices used to infer lane orientation on bridge cells.
        :return: Boolean array indicating which targets are occupiable.
        :rtype: NDArray[bool]
        """
        count: int = env_idx.size
        result: NDArray[np.bool_] = np.zeros((count,), dtype=np.bool_)
        if count == 0:
            return result

        bridge_mask: NDArray[np.bool_] = self._bridges_mask[rows, cols]
        normal_mask: NDArray[np.bool_] = ~bridge_mask
        if np.any(normal_mask):
            nm: NDArray[np.intp] = np.where(normal_mask)[0]
            result[nm] = self._grid_codes[env_idx[nm], rows[nm], cols[nm]] == 0

        if np.any(bridge_mask):
            bm: NDArray[np.intp] = np.where(bridge_mask)[0]
            dir_sel: NDArray[np.integer] = dir_idx[bm]
            vertical: NDArray[np.bool_] = (dir_sel % 2) == 0
            horizontal: NDArray[np.bool_] = ~vertical

            target: NDArray[np.unsignedinteger]
            if np.any(vertical):
                vi: NDArray[np.intp] = bm[vertical]
                target = self._lane_v[env_idx[vi], rows[vi], cols[vi]]
                result[vi] = (target == 0) | (target == color_codes[vi])

            if np.any(horizontal):
                hi: NDArray[np.intp] = bm[horizontal]
                target = self._lane_h[env_idx[hi], rows[hi], cols[hi]]
                result[hi] = (target == 0) | (target == color_codes[hi])

        return result

    def _occupies_other_stack(
        self,
        env_idx: NDArray[np.intp],
        color_idx: NDArray[np.integer],
        head_idx: NDArray[np.integer],
        rows: NDArray[np.signedinteger],
        cols: NDArray[np.signedinteger],
    ) -> NDArray[np.bool_]:
        """Return mask of positions that appear in the opposite head's stack.

        The method checks for each target position whether it is present in the stack of the opposite head for the same
        color and environment.

        :param env_idx: Environment indices to check.
        :param color_idx: Color indices of the paths being checked.
        :param head_idx: Head indices (0 or 1) for the moving endpoints.
        :param rows: Row coordinates to check.
        :param cols: Column coordinates to check.
        :return: Boolean array where ``True`` indicates that the position exists in the opposite head's stack.
        :rtype: NDArray[bool]
        """
        if env_idx.size == 0:
            return np.zeros((0,), dtype=np.bool_)

        other_head: NDArray[np.integer] = 1 - head_idx
        row_idx: NDArray[np.intp] = rows.astype(np.intp, copy=False)
        col_idx: NDArray[np.intp] = cols.astype(np.intp, copy=False)
        return self._arm_presence[env_idx, color_idx, other_head, row_idx, col_idx]

    def _lane_codes(
        self, dir_idx: NDArray[np.integer], rows: NDArray[np.integer], cols: NDArray[np.integer]
    ) -> NDArray[np.uint8]:
        """Return lane orientation codes for the specified cells.

        For non-bridge cells the method returns :data:`LANE_NORMAL`. For bridge cells it returns :data:`LANE_VERTICAL`
        for vertical movement and :data:`LANE_HORIZONTAL` for horizontal movement.

        :param dir_idx: Direction indices where ``0`` is up, ``1`` is right, ``2`` is down, and ``3`` is left.
        :param rows: Row coordinates of the cells.
        :param cols: Column coordinates of the cells.
        :return: Array of lane codes for each provided position.
        :rtype: NDArray[np.uint8]
        """
        codes: NDArray[np.uint8] = np.full(dir_idx.shape, LANE_NORMAL, dtype=np.uint8)
        bridge_mask: NDArray[np.bool_] = self._bridges_mask[rows, cols]
        if np.any(bridge_mask):
            bm: NDArray[np.intp] = np.where(bridge_mask)[0]
            codes[bm] = np.where((dir_idx[bm] % 2) == 0, LANE_VERTICAL, LANE_HORIZONTAL)
        return codes

    # State predicates

    def _update_heads(self) -> None:
        """Update head positions so they reflect the current stack tops.

        The method computes the top index for each stack and assigns the coordinates at that index into the
        :attr:`_heads` array.
        """
        top_idx: NDArray[np.signedinteger] = np.clip(self._stack_len - 1, 0, self._stack_capacity - 1)
        rows: NDArray[np.signedinteger] = np.take_along_axis(self._stack_rows, top_idx[..., None], axis=3)[..., 0]
        cols: NDArray[np.signedinteger] = np.take_along_axis(self._stack_cols, top_idx[..., None], axis=3)[..., 0]
        self._heads[..., 0] = rows
        self._heads[..., 1] = cols

    def _recompute_closed(self, colors: NDArray[np.integer] | None = None) -> None:
        """Recompute connectivity status for the provided color indices.

        When ``colors`` is ``None`` the method recomputes connectivity for all colors. Connectivity is stored in the
        :attr:`_closed` array.

        :param colors: Optional array of color indices to check.
        :type colors: NDArray[np.integer] | None
        """
        color_indices: NDArray[np.integer] = (
            np.arange(self._num_colors, dtype=np.int32)
            if colors is None
            else np.unique(colors).astype(np.int32, copy=False)
        )
        for ci in color_indices:
            ci_int: int = int(ci)
            for env in range(self._num_envs):
                self._closed[env, ci_int] = self._is_color_connected(env, ci_int)

    def _is_color_connected(self, env: int, color_index: int) -> bool:
        """Return whether both endpoints of ``color_index`` are connected in environment ``env``."""
        color_code: int = color_index + 1
        start_r: int = int(self._endpoints[color_index, 0, 0])
        start_c: int = int(self._endpoints[color_index, 0, 1])
        goal_r: int = int(self._endpoints[color_index, 1, 0])
        goal_c: int = int(self._endpoints[color_index, 1, 1])

        if start_r == goal_r and start_c == goal_c:
            return True

        if not (
            self._cell_has_color(env, start_r, start_c, color_code)
            and self._cell_has_color(env, goal_r, goal_c, color_code)
        ):
            return False

        visited: NDArray[np.bool_] = self._visit_buffer
        visited.fill(False)
        queue: deque[Coord] = self._bfs_queue
        queue.clear()
        queue.append((start_r, start_c))
        visited[start_r, start_c] = True

        while queue:
            r, c = queue.popleft()
            if r == goal_r and c == goal_c:
                return True

            for dr, dc in self._dirs:
                nr: int = r + int(dr)
                nc: int = c + int(dc)
                if not (0 <= nr < self._height and 0 <= nc < self._width):
                    continue
                if visited[nr, nc]:
                    continue
                if self._cell_has_color(env, nr, nc, color_code):
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        return False

    def _compute_solved_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask of environments that satisfy solution rules.

        In path mode an environment is solved when all colors are connected and, if :attr:`variant.must_fill` is true,
        when all cells are filled as defined by :meth:`_all_filled`. In cell switching mode the method delegates to
        :meth:`_validate_cell_switch_solution`.

        :return: Boolean array where ``True`` indicates a solved environment.
        :rtype: NDArray[bool]
        """
        if self.variant.cell_switching_mode:
            return self._validate_cell_switch_solution()

        closed_all: NDArray[np.bool_] = np.all(self._closed, axis=1)
        if not self.variant.must_fill:
            return closed_all

        return closed_all & self._all_filled()

    def _all_filled(self) -> NDArray[np.bool_]:
        """Return a mask indicating environments where every cell is occupied.

        Ensures non-bridge cells hold a nonzero grid code and that each bridge cell has at least one occupied lane.

        :return: Boolean array indicating which environments are fully filled.
        :rtype: NDArray[bool]
        """
        normal_ok: NDArray[np.bool_] = np.all((self._grid_codes != 0) | self._bridges_mask, axis=(1, 2))
        bridge_ok: NDArray[np.bool_] = np.all(
            (~self._bridges_mask) | ((self._lane_v != 0) | (self._lane_h != 0)), axis=(1, 2)
        )
        return normal_ok & bridge_ok

    def _validate_cell_switch_solution(self) -> NDArray[np.bool_]:
        """Validate solutions according to cell switching rules for each env.

        The method enforces that each color forms a connected path between its endpoints, that endpoints have exactly
        one same-color neighbor, and that interior cells have exactly two same-color neighbors.
        If :attr:`variant.must_fill` is true the method requires that all cells are filled using :meth:`_all_filled`.

        :return: Boolean array indicating which environments are valid solutions under cell switching rules.
        :rtype: NDArray[bool]
        """
        solved: NDArray[np.bool_] = np.ones((self._num_envs,), dtype=np.bool_)
        if self.variant.must_fill:
            filled: NDArray[np.bool_] = self._all_filled()
            solved &= filled

        dirs: NDArray[np.intp] = self._dirs
        bridge_mask_global: NDArray[np.bool_] = self._bridges_mask
        for env in range(self._num_envs):
            if not solved[env]:
                continue

            for ci in range(self._num_colors):
                color_code: int = ci + 1
                ep0_r: int = self._endpoints[ci, 0, 0]
                ep0_c: int = self._endpoints[ci, 0, 1]
                ep1_r: int = self._endpoints[ci, 1, 0]
                ep1_c: int = self._endpoints[ci, 1, 1]

                visited_map: NDArray[np.bool_] = self._visit_buffer
                visited_map.fill(False)
                queue: deque[Coord] = self._bfs_queue
                queue.clear()
                queue.append((ep0_r, ep0_c))
                visited_map[ep0_r, ep0_c] = True

                while queue:
                    r, c = queue.popleft()
                    if r == ep1_r and c == ep1_c:
                        break
                    for dr, dc in dirs:
                        nr: int = r + int(dr)
                        nc: int = c + int(dc)
                        if not (0 <= nr < self._height and 0 <= nc < self._width):
                            continue
                        if visited_map[nr, nc]:
                            continue
                        if self._cell_has_color(env, nr, nc, color_code):
                            visited_map[nr, nc] = True
                            queue.append((nr, nc))

                if not visited_map[ep1_r, ep1_c]:
                    solved[env] = False
                    break

                color_mask: NDArray[np.bool_] = self._visit_buffer_alt
                color_mask.fill(False)
                grid_env: NDArray[np.unsignedinteger] = self._grid_codes[env]
                lane_v_env: NDArray[np.unsignedinteger] = self._lane_v[env]
                lane_h_env: NDArray[np.unsignedinteger] = self._lane_h[env]
                non_bridge_mask: NDArray[np.bool_] = (~bridge_mask_global) & (grid_env == color_code)
                color_mask[non_bridge_mask] = True
                if np.any(bridge_mask_global):
                    color_mask[(bridge_mask_global) & (lane_v_env == color_code)] = True
                    color_mask[(bridge_mask_global) & (lane_h_env == color_code)] = True

                if np.any(color_mask & ~visited_map):
                    solved[env] = False
                    break

                coords: NDArray[np.intp] = np.argwhere(color_mask)
                for coord in coords:
                    r = int(coord[0])
                    c = int(coord[1])
                    neighbour_count: int = 0
                    for dr, dc in dirs:
                        nr = r + int(dr)
                        nc = c + int(dc)
                        if not (0 <= nr < self._height and 0 <= nc < self._width):
                            continue
                        if color_mask[nr, nc]:
                            neighbour_count += 1

                    if (r == ep0_r and c == ep0_c) or (r == ep1_r and c == ep1_c):
                        if neighbour_count != 1:
                            solved[env] = False
                            break
                    elif neighbour_count != 2:
                        solved[env] = False
                        break

                if not solved[env]:
                    break
        return solved

    def _cell_has_color(self, env: int, row: int, col: int, color_code: int) -> bool:
        """Return whether the given cell contains the specified color code.

        For non-bridge cells the method tests the regular grid code. For bridge cells the method tests both vertical and
        horizontal lane arrays.

        :param env: Environment index.
        :param row: Row coordinate of the cell.
        :param col: Column coordinate of the cell.
        :param color_code: Color code to test for.
        :return: ``True`` when the cell contains ``color_code``.
        :rtype: bool
        """
        if not self._bridges_mask[row, col]:
            return self._grid_codes[env, row, col] == color_code

        return (self._lane_v[env, row, col] == color_code) or (self._lane_h[env, row, col] == color_code)

    # Action mask & info

    def _compute_action_mask(self) -> NDArray[np.uint8]:
        """Compute the binary action mask for all environments.

        When :attr:`variant.cell_switching_mode` is true the method returns a replicated static mask built by
        :meth:`_build_cell_switch_mask`. In path mode the method examines head positions, previous stack entries, other
        head positions, and occupancy rules enforced by :meth:`_can_occupy_targets` to determine valid moves.

        :return: Binary array of shape ``(num_envs, action_space_size)`` where ``1`` indicates a permitted action.
        :rtype: NDArray[np.uint8]
        """
        if self.variant.cell_switching_mode:
            assert self._static_cell_mask is not None
            return np.repeat(self._static_cell_mask[np.newaxis, :], self._num_envs, axis=0)

        mask: NDArray[np.uint8] = self._path_action_mask
        mask.fill(0)
        env_idx: NDArray[np.intp] = self._env_indices
        allow_buf: NDArray[np.bool_] = self._allow_buffer
        apply_buf: NDArray[np.bool_] = self._apply_buffer
        for ci in range(self._num_colors):
            color_code: int = ci + 1
            for hi in (0, 1):
                head: NDArray[np.signedinteger] = self._heads[:, ci, hi]
                head_rows: NDArray[np.signedinteger] = head[:, 0]
                head_cols: NDArray[np.signedinteger] = head[:, 1]
                stack_len: NDArray[np.signedinteger] = self._stack_len[:, ci, hi]
                has_prev: NDArray[np.bool_] = stack_len >= 2
                prev_idx: NDArray[np.signedinteger] = np.clip(stack_len - 2, 0, self._stack_capacity - 1)
                env_index: NDArray[np.intp] = env_idx
                # Use per-environment indexing to select the correct previous positions
                prev_rows: NDArray[np.signedinteger] = self._stack_rows[env_index, ci, hi, prev_idx]
                prev_cols: NDArray[np.signedinteger] = self._stack_cols[env_index, ci, hi, prev_idx]
                other_head: NDArray[np.signedinteger] = self._heads[:, ci, 1 - hi]
                other_ep: NDArray[np.signedinteger] = self._endpoints[ci, 1 - hi]
                other_len: NDArray[np.signedinteger] = self._stack_len[:, ci, 1 - hi]
                for d in range(self._num_dirs):
                    action_index: int = ci * self._actions_per_color + hi * self._num_dirs + d
                    delta: NDArray[np.signedinteger] = self._dirs[d]
                    target_r: NDArray[np.signedinteger] = head_rows + delta[0]
                    target_c: NDArray[np.signedinteger] = head_cols + delta[1]
                    in_bounds: NDArray[np.bool_] = (
                        (target_r >= 0) & (target_r < self._height) & (target_c >= 0) & (target_c < self._width)
                    )
                    allow_buf.fill(False)
                    backtrack: NDArray[np.bool_] = (
                        in_bounds & has_prev & (target_r == prev_rows) & (target_c == prev_cols)
                    )
                    allow_buf |= backtrack
                    join_head: NDArray[np.bool_] = (
                        in_bounds & (target_r == other_head[:, 0]) & (target_c == other_head[:, 1])
                    )
                    allow_buf |= join_head
                    join_endpoint: NDArray[np.bool_] = (
                        in_bounds & (target_r == other_ep[0]) & (target_c == other_ep[1]) & (other_len == 1)
                    )
                    allow_buf |= join_endpoint
                    occupy_mask: NDArray[np.bool_] = in_bounds & ~(backtrack | join_head | join_endpoint)
                    if np.any(occupy_mask):
                        subset = np.where(occupy_mask)[0]
                        rows: NDArray[np.signedinteger] = target_r[subset]
                        cols: NDArray[np.signedinteger] = target_c[subset]
                        color_codes: NDArray[np.unsignedinteger] = np.full(
                            rows.shape, color_code, dtype=self._color_code_dtype
                        )
                        dir_idx: NDArray[np.unsignedinteger] = np.full(rows.shape, d, dtype=self._dir_dtype)
                        can_occupy: NDArray[np.bool_] = self._can_occupy_targets(
                            env_idx[subset], rows, cols, color_codes, dir_idx
                        )
                        if np.any(can_occupy):
                            occ_subset: NDArray[np.intp] = subset[np.where(can_occupy)[0]]
                            interior: NDArray[np.bool_] = self._occupies_other_stack(
                                env_idx[occ_subset],
                                np.full(occ_subset.shape, ci, dtype=np.int32),
                                np.full(occ_subset.shape, hi, dtype=np.int32),
                                target_r[occ_subset],
                                target_c[occ_subset],
                            )
                            apply_buf.fill(False)
                            apply_buf[occ_subset] = ~interior
                            allow_buf |= apply_buf

                    mask[:, action_index] = allow_buf

        return mask.copy()

    def _build_cell_switch_mask(self) -> NDArray[np.uint8]:
        """Build a static action mask used when cell switching mode is active.

        Endpoint cells are excluded from writable targets. The mask encodes valid ``(row, col, color)`` combinations as
        a flat array of length :attr:`_cell_action_size`.

        :return: Binary array of length ``_cell_action_size`` where ``1`` indicates a valid action.
        :rtype: NDArray[np.uint8]
        """
        if self._cell_action_size == 0:
            return np.zeros((0,), dtype=np.uint8)

        allowed: NDArray[np.bool_] = (~self._endpoint_mask).reshape(-1)
        return np.repeat(allowed.astype(np.uint8, copy=False), self._cell_action_stride).astype(np.uint8, copy=False)

    def _build_info(
        self,
        *,
        action_mask: NDArray[np.uint8],
        solved: NDArray[np.bool_] | None = None,
        deadlocked: NDArray[np.bool_] | None = None,
    ) -> InfoDict:
        """Build the info dictionary returned by :meth:`step` and :meth:`reset`.

        The dictionary contains the current action mask, step counters, connection status per color, the level
        identifier, and flags for solved and deadlocked states. When ``solved`` or ``deadlocked`` are not provided the
        method computes them from the current state.

        :param action_mask: Binary action mask for each environment.
        :param solved: Optional precomputed solved mask for each environment.
        :param deadlocked: Optional precomputed deadlocked mask for each environment.
        :return: Dictionary with keys ``'action_mask'``, ``'steps'``, ``'connected'``, ``'level_id'``, ``'solved'``, and
            ``'deadlocked'``.
        :rtype: InfoDict
        """
        solved_final: NDArray[np.bool_] = self._compute_solved_mask() if solved is None else solved
        deadlocked_final: NDArray[np.bool_] = (
            (~solved_final) & (action_mask.sum(axis=1) == 0) if deadlocked is None else deadlocked
        )

        return {
            "action_mask": action_mask,
            "steps": self._step_count.copy(),
            "connected": self._closed,
            "level_id": np.full((self._num_envs,), self.level_id, dtype=str),
            "solved": solved_final.astype(bool),
            "deadlocked": deadlocked_final.astype(bool),
        }

    # Rendering

    def _render_rgb(self) -> ObsType:
        """Render an RGB image for each environment and return as an array.

        The method maps color codes to palette RGB values, blends bridge lane colors when both lanes are occupied, and
        applies endpoint styling according to :attr:`_render_cfg`.

        :return: Array of shape ``(num_envs, height, width, 3)`` with ``uint8`` RGB images.
        :rtype: ObsType
        """
        img: ObsType = np.zeros((self._num_envs, self._height, self._width, 3), dtype=np.uint8)
        # Apply background color per cell before filling colors
        bg: NDArray[np.uint8] = np.array(self._render_cfg.grid_background_color, dtype=np.uint8)
        if not np.all(bg == 0):
            img[...] = bg
        normal_mask: NDArray[np.bool_] = (self._grid_codes > 0) & (~self._bridges_mask)
        if np.any(normal_mask):
            colors: NDArray[np.unsignedinteger] = self._grid_codes[normal_mask] - 1
            img[normal_mask] = self._palette[colors]

        if np.any(self._bridges_mask):
            bridge_expanded: NDArray[np.bool_] = self._bridges_mask[np.newaxis, :, :]
            flat: NDArray[np.uint8] = img[bridge_expanded]
            v_codes: NDArray[np.unsignedinteger] = self._lane_v[bridge_expanded]
            h_codes: NDArray[np.unsignedinteger] = self._lane_h[bridge_expanded]
            both: NDArray[np.bool_] = (v_codes > 0) & (h_codes > 0)
            only_v: NDArray[np.bool_] = (v_codes > 0) & (h_codes == 0)
            only_h: NDArray[np.bool_] = (h_codes > 0) & (v_codes == 0)
            none: NDArray[np.bool_] = (v_codes == 0) & (h_codes == 0)
            flat[none] = 0
            if np.any(only_v):
                flat[only_v] = self._palette[v_codes[only_v] - 1]
            if np.any(only_h):
                flat[only_h] = self._palette[h_codes[only_h] - 1]
            if np.any(both):
                flat[both] = (self._palette[v_codes[both] - 1] + self._palette[h_codes[both] - 1]) // 2

        # Apply endpoint styling based on render config
        for ci in range(self._num_colors):
            ep0: NDArray[np.int32] = self._endpoints[ci, 0]
            ep1: NDArray[np.int32] = self._endpoints[ci, 1]

            if self._render_cfg.endpoint_border_thickness > 0:
                # Render endpoints with border: use base color (border drawn externally by viewer)
                img[:, ep0[0], ep0[1]] = self._palette[ci]
                img[:, ep1[0], ep1[1]] = self._palette[ci]
            elif self._render_cfg.connection_color_adjustment != 0:
                # Apply color adjustment when no border is used
                adjusted: NDArray[np.uint8] = np.clip(
                    self._palette[ci].astype(np.int16) + self._render_cfg.connection_color_adjustment,
                    0,
                    255,
                    dtype=np.uint8,
                )
                img[:, ep0[0], ep0[1]] = adjusted
                img[:, ep1[0], ep1[1]] = adjusted

        # Upscale if needed using efficient numpy repeat
        if self._pixels_per_cell_h > 1 or self._pixels_per_cell_w > 1:
            img = np.repeat(img, self._pixels_per_cell_h, axis=1)
            img = np.repeat(img, self._pixels_per_cell_w, axis=2)

            # Draw endpoint borders at the correct scale after gridlines and before labels
            if (
                self._render_cfg.endpoint_border_thickness > 0
                and self._pixels_per_cell_h >= 2
                and self._pixels_per_cell_w >= 2
            ):
                border_color: NDArray[np.uint8] = np.array(self._render_cfg.endpoint_border_color, dtype=np.uint8)
                thickness: int = int(self._render_cfg.endpoint_border_thickness)
                grid_thickness_for_inset: int = (
                    max(1, int(self._render_cfg.gridline_thickness))
                    if self._render_cfg.gridline_color is not None
                    else 0
                )

                for ci in range(self._num_colors):
                    ep0 = self._endpoints[ci, 0]
                    ep1 = self._endpoints[ci, 1]

                    for ep_r, ep_c in [(ep0[0], ep0[1]), (ep1[0], ep1[1])]:
                        # Convert to Python int to avoid overflow in multiplication
                        r0: int = int(ep_r) * self._pixels_per_cell_h
                        r1: int = r0 + self._pixels_per_cell_h
                        c0: int = int(ep_c) * self._pixels_per_cell_w
                        c1: int = c0 + self._pixels_per_cell_w

                        # Compute per-side overlaps for precise insetting to avoid off-by-one on right/bottom
                        if grid_thickness_for_inset > 0:
                            hpx: int = img.shape[1]
                            wpx: int = img.shape[2]

                            # Left gridline at x = c0
                            x_left: int = c0
                            x0g_left: int = x_left - (grid_thickness_for_inset // 2)
                            x1g_left: int = x0g_left + grid_thickness_for_inset
                            x0g_left_clipped: int = max(0, x0g_left)
                            x1g_left_clipped: int = min(wpx, x1g_left)
                            left_overlap: int = max(0, min(x1g_left_clipped, c1) - max(x0g_left_clipped, c0))

                            # Right gridline at x = c1
                            x_right: int = c1
                            x0g_right: int = x_right - (grid_thickness_for_inset // 2)
                            x1g_right: int = x0g_right + grid_thickness_for_inset
                            x0g_right_clipped: int = max(0, x0g_right)
                            x1g_right_clipped: int = min(wpx, x1g_right)
                            right_overlap: int = max(0, min(x1g_right_clipped, c1) - max(x0g_right_clipped, c0))

                            # Top gridline at y = r0
                            y_top: int = r0
                            y0g_top: int = y_top - (grid_thickness_for_inset // 2)
                            y1g_top: int = y0g_top + grid_thickness_for_inset
                            y0g_top_clipped: int = max(0, y0g_top)
                            y1g_top_clipped: int = min(hpx, y1g_top)
                            top_overlap: int = max(0, min(y1g_top_clipped, r1) - max(y0g_top_clipped, r0))

                            # Bottom gridline at y = r1
                            y_bottom: int = r1
                            y0g_bottom: int = y_bottom - (grid_thickness_for_inset // 2)
                            y1g_bottom: int = y0g_bottom + grid_thickness_for_inset
                            y0g_bottom_clipped: int = max(0, y0g_bottom)
                            y1g_bottom_clipped: int = min(hpx, y1g_bottom)
                            bottom_overlap: int = max(0, min(y1g_bottom_clipped, r1) - max(y0g_bottom_clipped, r0))
                        else:
                            left_overlap = right_overlap = top_overlap = bottom_overlap = 0

                        # Interior box after subtracting exact gridline overlaps on each side
                        ir0: int = r0 + top_overlap
                        ir1: int = r1 - bottom_overlap
                        ic0: int = c0 + left_overlap
                        ic1: int = c1 - right_overlap
                        ih: int = max(0, ir1 - ir0)
                        iw: int = max(0, ic1 - ic0)
                        if ih <= 0 or iw <= 0:
                            continue

                        t: int = min(thickness, ih, iw)
                        if t <= 0:
                            continue

                        # Draw borders for all environments (vectorized along batch dimension) on the inner side
                        # Top border
                        img[:, ir0 : ir0 + t, ic0:ic1] = border_color
                        # Bottom border
                        img[:, ir1 - t : ir1, ic0:ic1] = border_color
                        # Left border
                        img[:, ir0:ir1, ic0 : ic0 + t] = border_color
                        # Right border
                        img[:, ir0:ir1, ic1 - t : ic1] = border_color

        # Render endpoint numbers last if enabled and there are enough pixels-per-cell to draw cleanly
        if self._render_cfg.show_endpoint_numbers and (self._pixels_per_cell_h >= 10 and self._pixels_per_cell_w >= 10):
            specs: list[tuple[str, Coord, RGBInt, int]] = build_endpoint_labels(
                endpoints=self._endpoints,
                pixels_per_cell_h=self._pixels_per_cell_h,
                pixels_per_cell_w=self._pixels_per_cell_w,
                min_scale=self._render_cfg.number_font_min_scale,
                max_scale=self._render_cfg.number_font_max_scale,
                gridline_thickness=self._render_cfg.gridline_thickness or 0,
            )
            # Render numbers for each environment in the batch
            num_color: RGBInt = (
                self._render_cfg.number_font_color[0],
                self._render_cfg.number_font_color[1],
                self._render_cfg.number_font_color[2],
            )
            for env_idx in range(self._num_envs):
                for text, center, _color, scale in specs:
                    render_bitmap_text_centered(
                        text,
                        center,
                        num_color,
                        img[env_idx],
                        scale,
                        self._render_cfg.number_font_border_color,
                        self._render_cfg.number_font_border_thickness,
                    )

        # Draw gridlines last so they appear on top of numbers, endpoint borders, and other content
        if self._render_cfg.gridline_color is not None and (
            self._pixels_per_cell_h >= 2 and self._pixels_per_cell_w >= 2
        ):
            gl_color: NDArray[np.uint8] = np.array(self._render_cfg.gridline_color, dtype=np.uint8)
            grid_thickness: int = max(1, self._render_cfg.gridline_thickness)
            hpx = img.shape[1]
            wpx = img.shape[2]
            cell_h: int = self._pixels_per_cell_h
            cell_w: int = self._pixels_per_cell_w

            # Vertical lines
            for c in range(self._width + 1):
                x: int = c * cell_w
                x0: int = max(0, x - grid_thickness // 2)
                x1: int = min(wpx, x0 + grid_thickness)
                img[:, :, x0:x1, :] = gl_color

            # Horizontal lines
            for r in range(self._height + 1):
                y: int = r * cell_h
                y0: int = max(0, y - grid_thickness // 2)
                y1: int = min(hpx, y0 + grid_thickness)
                img[:, y0:y1, :, :] = gl_color

        return img
