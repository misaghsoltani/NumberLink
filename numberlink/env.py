"""NumberLink environment implementation.

Provide the :class:`numberlink.env.NumberLinkRGBEnv` environment, a Gymnasium-compatible RGB NumberLink
environment. Observations are RGB images with shape ``(H, W, 3)`` and ``dtype==numpy.uint8``. The discrete action
space encodes a color, one of two heads for that color, and a direction. When the
:attr:`numberlink.config.VariantConfig.cell_switching_mode` variant is enabled, actions encode a ``(row, col, color)``
assignment.

See :mod:`numberlink.level_setup` for level construction utilities and :mod:`numberlink.config` for configuration
structures referenced by this module.

.. versionadded:: 1.0

.. note::
    When Sphinx renders documentation with ``autodoc_typehints`` enabled, type information is taken from function
    signatures and class attributes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import RenderFrame
import numpy as np

from numberlink.config import RenderConfig, RewardConfig, VariantConfig
from numberlink.level_setup import LevelTemplate

from .level_setup import build_level_template
from .number_render import build_endpoint_labels, render_bitmap_text_centered
from .types import ActType, ObsType, select_unsigned_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Any, TypeAlias

    from numpy.typing import NDArray

    from .config import GeneratorConfig, RenderConfig, RewardConfig, VariantConfig
    from .level_setup import LevelTemplate
    from .types import CellLane, Coord, Lane, RenderMode, RGBInt

    InfoValue: TypeAlias = NDArray[np.uint8] | int | NDArray[np.bool_] | str | None | bool
    InfoDict: TypeAlias = dict[str, InfoValue]


class NumberLinkRGBEnv(gym.Env[ObsType, ActType]):
    """NumberLink environment.

    Expose observations as RGB arrays with shape ``(H, W, 3)`` and ``dtype==numpy.uint8``. Action encoding depends on
    the active variant. In path-building mode actions encode ``(color, head, direction)``. In cell-switching mode
    actions encode a ``(row, col, color)`` assignment.

    The environment implements bridge cells with independent lanes, optional diagonal movement, and an optional
    requirement that all cells be filled for a solution to be considered valid.

    Interact with the environment via :meth:`numberlink.env.NumberLinkRGBEnv.reset`,
    :meth:`numberlink.env.NumberLinkRGBEnv.step`, and :meth:`numberlink.env.NumberLinkRGBEnv.render`.

    Internal NumPy arrays expose state for read only access. Do not mutate arrays returned by public methods.
    """

    metadata: dict[str, list[str] | int] = {"render_modes": ["rgb_array", "ansi", "human"], "render_fps": 30}

    _DIRS4: NDArray[np.int8] = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
    _DIRS8: NDArray[np.int8] = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]], dtype=np.int8
    )

    def __init__(
        self,
        grid: Sequence[str] | None = None,
        *,
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
        """Initialize the environment.

        Load the provided ``grid`` or generate a level using the provided ``generator`` configuration and apply any
        configuration overrides passed to this constructor. The environment uses explicit configuration values when
        corresponding overrides are supplied.

        :param grid: Optional iterable of strings representing the level grid rows.
        :type grid: Sequence[str] or None
        :param render_mode: Optional render mode forwarded to :meth:`numberlink.env.NumberLinkRGBEnv.render`.
        :type render_mode: RenderMode or None
        :param level_id: Optional identifier for a predefined level.
        :type level_id: str or None
        :param variant: Optional :class:`numberlink.config.VariantConfig` to override template variant.
        :type variant: VariantConfig or None
        :param bridges: Optional iterable of bridge coordinates as ``(row, col)`` pairs.
        :type bridges: Iterable[Coord] or None
        :param generator: Optional :class:`numberlink.config.GeneratorConfig` to generate a level.
        :type generator: GeneratorConfig or None
        :param reward_config: Optional :class:`numberlink.config.RewardConfig` override.
        :type reward_config: RewardConfig or None
        :param render_config: Optional :class:`numberlink.config.RenderConfig` override.
        :type render_config: RenderConfig or None
        :param step_limit: Optional maximum number of steps before truncation. If ``None``, a default is used.
        :type step_limit: int or None
        :param palette: Optional mapping of letter to RGB tuple used for rendering.
        :type palette: dict[str, RGBInt] or None

        :raises ValueError: If configured render dimensions are smaller than the grid dimensions.
        :raises RuntimeError: If the generator configuration is invalid for level generation.
        :seealso: :func:`numberlink.level_setup.build_level_template`
        """
        super().__init__()
        self.render_mode = render_mode
        self._source_grid: list[str] | None = None
        if grid is not None:
            self._source_grid = [str(row) for row in grid]
        self._source_bridges: tuple[Coord, ...] | None = None
        if bridges is not None:
            self._source_bridges = tuple((int(coord[0]), int(coord[1])) for coord in bridges)
        self._source_level_id: str | None = level_id
        self._variant_override: VariantConfig | None = variant
        self._reward_config_override: RewardConfig | None = reward_config
        self._render_config_override: RenderConfig | None = render_config
        self._palette_override: dict[str, RGBInt] | None = dict(palette) if palette is not None else None
        self._generator_config: GeneratorConfig | None = generator
        self._step_limit_override: int | None = step_limit

        template: LevelTemplate = build_level_template(
            grid=self._source_grid,
            level_id=self._source_level_id,
            variant=self._variant_override,
            bridges=self._source_bridges,
            generator=self._generator_config,
            reward_config=self._reward_config_override,
            render_config=self._render_config_override,
            palette=self._palette_override,
            solution=solution,
        )
        self._apply_template(template)

        # Internal state
        self._grid_codes: NDArray[np.unsignedinteger]
        self._lane_v: NDArray[np.unsignedinteger]
        self._lane_h: NDArray[np.unsignedinteger]
        self._heads: list[list[Coord]]
        self._stacks: list[list[list[CellLane]]]
        self._closed: NDArray[np.bool_] = np.zeros(self.num_colors, dtype=np.bool_)
        self._steps: int = 0
        # Convert coordinate solution to action sequence
        self._solution: list[ActType] | None = self._compute_solution_actions()

    def _apply_template(self, template: LevelTemplate) -> None:
        """Load derived attributes from a precomputed template."""
        self._template: LevelTemplate = template
        self.variant: VariantConfig = template.variant
        self._reward_cfg: RewardConfig = template.reward_config
        self._render_cfg: RenderConfig = template.render_config
        self.level_id: str | None = template.level_id

        self._raw_grid: list[str] = list(template.raw_grid)
        self.H: int = template.height
        self.W: int = template.width

        self._bridges: NDArray[np.bool_] = template.bridges_mask.copy()
        self._letters: list[str] = list(template.letters)
        self.num_colors: int = template.num_colors
        self._palette_map: dict[str, RGBInt] = dict(template.palette_map)
        self._palette_arrays: list[NDArray[np.uint8]] = [arr.copy() for arr in template.palette_arrays]
        self._palette_stack: NDArray[np.uint8] = np.stack(self._palette_arrays, axis=0)
        self._endpoint_base_colors: NDArray[np.uint8] = np.repeat(self._palette_stack, 2, axis=0)
        if self._render_cfg.connection_color_adjustment != 0:
            adjusted_stack: NDArray[np.uint8] = np.clip(
                self._palette_stack.astype(np.int16) + self._render_cfg.connection_color_adjustment, 0, 255
            ).astype(np.uint8)
            self._endpoint_adjusted_colors: NDArray[np.uint8] | None = np.repeat(adjusted_stack, 2, axis=0)
        else:
            self._endpoint_adjusted_colors = None
        self._grid_background_color: NDArray[np.uint8] = np.array(
            self._render_cfg.grid_background_color, dtype=np.uint8
        )
        self._color_code_dtype: type[np.unsignedinteger] = select_unsigned_dtype(self.num_colors)

        self._solution_coords: list[list[Coord]] | None = template.solution

        self._endpoints_array: NDArray[np.int16] = template.endpoints.astype(np.int16, copy=True)
        self._endpoints: list[tuple[Coord, Coord]] = [
            ((pair[0][0], pair[0][1]), (pair[1][0], pair[1][1])) for pair in self._endpoints_array
        ]
        self._endpoint_mask: NDArray[np.bool_] = np.zeros((self.H, self.W), dtype=np.bool_)
        self._endpoint_mask[self._endpoints_array[:, :, 0], self._endpoints_array[:, :, 1]] = True
        self._flat_endpoint_mask: NDArray[np.bool_] = self._endpoint_mask.reshape(-1)

        bridge_rows: NDArray[np.intp]
        bridge_cols: NDArray[np.intp]
        bridge_rows, bridge_cols = np.nonzero(self._bridges)
        self._bridge_rows: NDArray[np.intp] = bridge_rows.astype(np.intp, copy=False)
        self._bridge_cols: NDArray[np.intp] = bridge_cols.astype(np.intp, copy=False)

        self._cell_switch_rows: NDArray[np.int16] | None = None
        self._cell_switch_cols: NDArray[np.int16] | None = None
        self._cell_switch_colors: NDArray[np.int16] | None = None
        self._cell_switch_mask: NDArray[np.uint8] | None = None
        if self.variant.cell_switching_mode:
            colors_plus_clear: int = self.num_colors + 1
            total_actions: int = template.cell_switch_action_space_size
            action_indices: NDArray[np.int32] = np.arange(total_actions, dtype=np.int32)
            cell_indices: NDArray[np.int32] = action_indices // colors_plus_clear
            self._cell_switch_rows = (cell_indices // self.W).astype(np.int16, copy=False)
            self._cell_switch_cols = (cell_indices % self.W).astype(np.int16, copy=False)
            self._cell_switch_colors = (action_indices % colors_plus_clear).astype(np.int16, copy=False)
            allowed_cells: NDArray[np.bool_] = ~self._flat_endpoint_mask
            self._cell_switch_mask = np.repeat(allowed_cells, colors_plus_clear).astype(np.uint8, copy=False)
            self._cell_switch_mask.flags.writeable = False

        self._dirs: NDArray[np.int8] = template.dirs.copy()
        self._num_dirs: int = template.num_dirs
        self._actions_per_color: int = template.actions_per_color
        self._dir_to_index: dict[Coord, int] = {(int(vec[0]), int(vec[1])): idx for idx, vec in enumerate(self._dirs)}

        self.action_space = spaces.Discrete(
            template.cell_switch_action_space_size if self.variant.cell_switching_mode else template.action_space_size
        )

        self._pixels_per_cell_h: int = 1
        self._pixels_per_cell_w: int = 1
        obs_height: int = self.H
        obs_width: int = self.W
        pixels_per_cell_h_candidate: int = 1
        pixels_per_cell_w_candidate: int = 1

        if self._render_cfg.render_height is not None:
            if self._render_cfg.render_height < self.H:
                raise ValueError(f"render_height ({self._render_cfg.render_height}) must be >= grid height ({self.H})")
            pixels_per_cell_h_candidate = self._render_cfg.render_height // self.H

        if self._render_cfg.render_width is not None:
            if self._render_cfg.render_width < self.W:
                raise ValueError(f"render_width ({self._render_cfg.render_width}) must be >= grid width ({self.W})")
            pixels_per_cell_w_candidate = self._render_cfg.render_width // self.W

        # Use the minimum of the two to ensure cells are always square
        pixels_per_cell: int = min(pixels_per_cell_h_candidate, pixels_per_cell_w_candidate)
        self._pixels_per_cell_h = pixels_per_cell
        self._pixels_per_cell_w = pixels_per_cell
        obs_height = self._pixels_per_cell_h * self.H
        obs_width = self._pixels_per_cell_w * self.W

        self.observation_space = spaces.Box(low=0, high=255, shape=(obs_height, obs_width, 3), dtype=np.uint8)

        # Allocate reusable buffers for performance critical paths
        path_action_size: int = self.num_colors * self._actions_per_color
        self._path_action_mask: NDArray[np.uint8] = np.zeros((path_action_size,), dtype=np.uint8)
        self._rgb_buffer: NDArray[np.uint8] = np.zeros((obs_height, obs_width, 3), dtype=np.uint8)
        bridge_count: int = self._bridge_rows.size
        self._bridge_color_buffer: NDArray[np.uint8] | None = (
            np.zeros((bridge_count, 3), dtype=np.uint8) if bridge_count > 0 else None
        )
        self._visited_buffer: NDArray[np.bool_] = np.zeros((self.H, self.W), dtype=np.bool_)
        self._visited_buffer_alt: NDArray[np.bool_] = np.zeros((self.H, self.W), dtype=np.bool_)
        self._bfs_queue: deque[Coord] = deque()
        self._arm_presence: NDArray[np.bool_] = np.zeros((self.num_colors, 2, self.H, self.W), dtype=np.bool_)

        self.max_steps: int = (
            self._step_limit_override if self._step_limit_override is not None else 10 * self.H * self.W
        )
        self.step_count_dtype: type[np.unsignedinteger] = select_unsigned_dtype(self.max_steps)

        self._solution = self._compute_solution_actions()

    def _compute_solution_actions(self) -> list[ActType] | None:
        """Convert a coordinate-based solution to a sequence of action indices.

        Return ``None`` if no coordinate solution is available or when the active variant uses
        :attr:`numberlink.config.VariantConfig.cell_switching_mode` and conversion is not applicable.

        :return: List of action indices that solve the puzzle, or ``None`` if unavailable.
        :rtype: list[ActType] or None
        """
        if self._solution_coords is None:
            return None

        actions: list[ActType] = []
        if self.variant.cell_switching_mode:
            # For cell-switching mode, convert coordinates to cell-switching actions

            for color_idx, path in enumerate(self._solution_coords):
                if len(path) <= 2:
                    continue  # Only endpoints, no intermediate cells

                # Skip first and last cells (endpoints)
                for coord in path[1:-1]:
                    action: ActType = self.encode_cell_switching_action(coord[0], coord[1], color_idx + 1)
                    actions.append(action)

            return actions if actions else None

        # For path-building mode, convert coordinate paths to movement actions
        actions = []

        for color_idx, path in enumerate(self._solution_coords):
            if len(path) <= 1:
                continue

            # Determine which endpoint is the start of this path
            ep0, ep1 = self._endpoints[color_idx]

            # Find which head to use (0 or 1) based on path start
            if path[0] == ep0:
                head_idx = 0
            elif path[0] == ep1:
                head_idx = 1
            else:
                # Path doesn't start at an endpoint, skip
                continue

            # Convert consecutive coordinates to directional actions
            for i in range(len(path) - 1):
                current: Coord = path[i]
                next_cell: Coord = path[i + 1]

                # Calculate direction
                dr: int = next_cell[0] - current[0]
                dc: int = next_cell[1] - current[1]

                dir_idx: int | None = self._dir_to_index.get((dr, dc))
                if dir_idx is None:
                    continue

                action_idx: ActType = color_idx * self._actions_per_color + head_idx * self._num_dirs + dir_idx
                actions.append(action_idx)

        return actions if actions else None

    # Gymnasium API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, InfoDict]:
        """Reset the environment to the initial state.

        :param seed: Optional seed used for RNG initialization.
        :type seed: int or None
        :param options: Options forwarded to the environment reset. This implementation does not use options.
        :type options: dict[str, Any] or None

        :return: Tuple of the initial observation and an info dictionary.

        The info dictionary contains the keys

            - ``action_mask`` (:class:`numpy.ndarray`): Binary action mask for valid actions
            - ``steps`` (:class:`int`): Number of steps taken, ``0`` at reset
            - ``connected`` (:class:`numpy.ndarray`): Boolean array indicating which colors are connected
            - ``level_id`` (:class:`str` or ``None``): Identifier of the loaded level when available
            - ``solved`` (:class:`bool`): ``False`` at reset unless the level is trivially solved
            - ``deadlocked`` (:class:`bool`): Whether the state is deadlocked

        :rtype: tuple[:class:`numpy.ndarray`, dict]

        .. note::

            This method calls :meth:`numberlink.env.NumberLinkRGBEnv.render` when ``render_mode`` is set to
            ``'ansi'`` or ``'human'``.
        """
        super().reset(seed=seed)

        self._grid_codes = np.zeros((self.H, self.W), dtype=self._color_code_dtype)
        self._lane_v = np.zeros((self.H, self.W), dtype=self._color_code_dtype)
        self._lane_h = np.zeros((self.H, self.W), dtype=self._color_code_dtype)

        # Initialize stacks & heads using precomputed endpoints
        self._stacks = []
        self._heads = []
        self._closed = np.zeros((self.num_colors,), dtype=np.bool_)
        self._arm_presence.fill(False)
        for ci, (ep0, ep1) in enumerate(self._endpoints):
            color_code: int = ci + 1
            self._stacks.append([[(*ep0, "n")], [(*ep1, "n")]])
            self._heads.append([ep0, ep1])
            self._occupy_cell(ep0, color_code, lane="n")
            self._occupy_cell(ep1, color_code, lane="n")
            self._arm_presence[ci, 0, ep0[0], ep0[1]] = True
            self._arm_presence[ci, 1, ep1[0], ep1[1]] = True

        if self.variant.cell_switching_mode:
            self._refresh_cell_switch_connections(range(self.num_colors))
        self._steps = 0

        obs: ObsType = self._render_rgb()
        action_mask: NDArray[np.uint8] = self._compute_action_mask()
        info: InfoDict = {
            "action_mask": action_mask,
            "steps": self._steps,
            "connected": self._closed,
            "level_id": self.level_id,
            "solved": False,
            "deadlocked": False,
        }
        if self.render_mode in {"ansi", "human"}:
            self.render()
        return obs, info

    def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, InfoDict]:
        """Apply an action and advance the environment state by one step.

        Decode and apply ``action``, update internal state, compute the reward for the step, and return the next
        observation and step information.

        :param action: Encoded action in the discrete action space
        :type action: ActType

        :return: Tuple ``(observation, reward, terminated, truncated, info)`` where the components are:

            - ``observation``: the new observation frame
            - ``reward``: the scalar reward for the step
            - ``terminated``: whether the episode ended due to success or deadlock
            - ``truncated``: whether the episode ended due to reaching ``max_steps``
            - ``info``: a dictionary with keys described in :meth:`numberlink.env.NumberLinkRGBEnv.reset`

        :rtype: tuple[ObsType, float, bool, bool, InfoDict]

        .. note::

            ``terminated`` is ``True`` when the puzzle is solved or when the environment is deadlocked and cannot
            continue.

        .. note::

            ``truncated`` is ``True`` when the environment reached ``max_steps``.
        """
        action_index: int = int(action)
        self._steps += 1
        reward: float = self._reward_cfg.step_penalty
        terminated: bool = False
        truncated: bool = self._steps >= self.max_steps

        connected_before: np.bool_ = np.sum(self._closed)
        valid: bool = self._apply_action(action_index)
        if not valid:
            reward += self._reward_cfg.invalid_penalty

        connected_after: int = int(np.sum(self._closed))
        if connected_after > connected_before:
            reward += self._reward_cfg.connect_bonus * (connected_after - connected_before)

        # Check win condition
        solved: bool = self._is_solved()
        if solved:
            terminated = True
            reward += self._reward_cfg.win_bonus

        # Compute action mask for next state
        action_mask: NDArray[np.uint8] = self._compute_action_mask()

        # Check deadlock: no valid moves available and not solved
        deadlocked: bool = self._is_deadlocked(action_mask, solved)
        if deadlocked:
            terminated = True
            # Negative reward for deadlock (failed to solve)
            reward += self._reward_cfg.invalid_penalty * 2

        obs: ObsType = self._render_rgb()
        info: InfoDict = {
            "action_mask": action_mask,
            "steps": self._steps,
            "connected": self._closed,
            "level_id": self.level_id,
            "solved": solved,
            "deadlocked": deadlocked,
        }
        if self.render_mode in {"ansi", "human"}:
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Render the current environment state according to :attr:`render_mode`.

        When :attr:`render_mode` is ``'rgb_array'`` return an image frame of type :class:`numpy.ndarray`. When
        :attr:`render_mode` is ``'ansi'`` or ``'human'`` return a textual representation. When :attr:`render_mode` is
        ``'human'`` and :attr:`numberlink.config.RenderConfig.print_text_in_human_mode` is ``True``, also print the
        textual representation to standard output.

        :return: A frame or textual rendering depending on the configured mode, or ``None`` when rendering is
            disabled
        :rtype: RenderFrame or list[RenderFrame] or None
        """
        if self.render_mode == "rgb_array":
            return cast(RenderFrame, self._render_rgb())

        if self.render_mode in {"ansi", "human"}:
            s: str = self._render_text()
            if self.render_mode == "human" and self._render_cfg.print_text_in_human_mode:
                print(s)
            return cast(RenderFrame, s)

        return None

    def close(self) -> None:
        """Close the environment and release any acquired resources.

        This method is present for API compatibility with Gymnasium. Implementations may be a no op for in-memory
        stateful environments.
        """
        pass

    def get_solution(self) -> list[ActType] | None:
        """Return the solution action sequence for the current level when available.

        The returned list contains action indices that, when applied in sequence via
        :meth:`numberlink.env.NumberLinkRGBEnv.step`, will produce a solved state. Return
        ``None`` if no solution is available.

        For path-building variants actions encode color, head, and direction. For cell-switching variants actions
        encode cell assignments.

        :return: List of action indices that solve the puzzle, or ``None`` if unavailable
        :rtype: list[ActType] or None
        """
        return self._solution

    def regenerate_level(self, seed: int | None = None) -> tuple[ObsType, InfoDict]:
        """Generate a new level using the stored generator configuration and reset the environment.

        :param seed: Optional seed forwarded to the generator and used by :meth:`numberlink.env.NumberLinkRGBEnv.reset`.
        :type seed: int or None
        :return: Observation and info dictionary produced by :meth:`numberlink.env.NumberLinkRGBEnv.reset`
        :rtype: tuple[ObsType, InfoDict]
        :raises RuntimeError: If the environment was not created with a generator configuration
        """
        if self._generator_config is None:
            raise RuntimeError("regenerate_level requires an environment created with a generator configuration")

        base_generator: GeneratorConfig = self._generator_config
        generator_seed: int | None = (
            seed if seed is not None else (None if base_generator.seed is None else base_generator.seed + 1)
        )

        new_generator: GeneratorConfig = replace(base_generator, seed=generator_seed)
        template: LevelTemplate = build_level_template(
            grid=self._source_grid,
            level_id=self._source_level_id,
            variant=self._variant_override,
            bridges=self._source_bridges,
            generator=new_generator,
            reward_config=self._reward_config_override,
            render_config=self._render_config_override,
            palette=self._palette_override,
        )
        self._generator_config = new_generator
        self._apply_template(template)
        return self.reset(seed=seed)

    def _metric(self, a: Coord, b: Coord) -> int:
        """Compute the distance between two grid coordinates using the active metric.

        Use Chebyshev distance when diagonal moves are allowed, otherwise use Manhattan distance.

        :param a: First coordinate as ``(row, col)``
        :type a: Coord
        :param b: Second coordinate as ``(row, col)``
        :type b: Coord

        :return: Distance between the two coordinates
        :rtype: int
        """
        dr: int = abs(a[0] - b[0])
        dc: int = abs(a[1] - b[1])

        if self.variant.allow_diagonal:
            return max(dr, dc)

        return dr + dc

    # Action helpers
    def _decode_action(self, a: int) -> RGBInt:
        """Decode a packed action index into ``(color_index, head_index, direction)``.

        Return ``(-1, -1, -1)`` when the supplied index is out of range.

        :param a: Encoded action index
        :type a: int
        :return: Decoded components as integers where each value is non-negative, or ``-1`` for invalid inputs
        :rtype: RGBInt
        """
        total_actions: int = self.num_colors * self._actions_per_color
        if not (0 <= a < total_actions):
            return -1, -1, -1

        color_index: int = a // self._actions_per_color
        r: int = a % self._actions_per_color
        head_index: int = r // self._num_dirs
        direction: int = r % self._num_dirs
        return color_index, head_index, direction

    def _decode_cell_switching_action(self, a: int) -> RGBInt:
        """Decode a cell-switching action into ``(row, col, color_value)``.

        Return ``(-1, -1, -1)`` when the action index is invalid. Color value ``0`` means clear the cell. Values
        ``1..N`` map to color indices ``0..N-1``.

        :param a: Encoded cell switching action index
        :type a: int ``(-1, -1, -1)`` if invalid
        :rtype: RGBInt
        """
        if self._cell_switch_rows is None or self._cell_switch_cols is None or self._cell_switch_colors is None:
            colors_plus_clear: int = self.num_colors + 1
            total_actions: int = self.H * self.W * colors_plus_clear
            if not (0 <= a < total_actions):
                return -1, -1, -1
            cell_idx: int = a // colors_plus_clear
            color_value: int = a % colors_plus_clear
            row: int = cell_idx // self.W
            col: int = cell_idx % self.W
            return row, col, color_value

        if not (0 <= a < self._cell_switch_rows.size):
            return -1, -1, -1

        return (self._cell_switch_rows[a], self._cell_switch_cols[a], self._cell_switch_colors[a])

    def encode_cell_switching_action(self, row: int, col: int, color_value: int) -> int:
        """Encode a ``(row, col, color_value)`` triple into a discrete action index.

        Color value ``0`` clears the cell. Color values ``1..N`` map to color indices ``1..N``.

        :param row: Row index of the target cell
        :type row: int
        :param col: Column index of the target cell
        :type col: int
        :param color_value: Color value to assign (``0`` means clear, ``1..N`` are colors)
        :type color_value: int
        :return: Encoded action index suitable for :meth:`numberlink.env.NumberLinkRGBEnv.step`
        :rtype: int
        :raises ValueError: If any component is out of the valid ranges
        """
        colors_plus_clear: int = self.num_colors + 1
        if not (0 <= row < self.H and 0 <= col < self.W and 0 <= color_value < colors_plus_clear):
            raise ValueError("Cell switching action components out of range")
        return (row * self.W + col) * colors_plus_clear + color_value

    def _apply_action(self, a: int) -> bool:
        """Apply an encoded action according to the active environment mode.

        Dispatch to :meth:`numberlink.env.NumberLinkRGBEnv._apply_cell_switching_action` when the environment is in
        cell-switching mode. Otherwise dispatch to :meth:`numberlink.env.NumberLinkRGBEnv._apply_path_action`.

        :param a: Encoded action index
        :type a: int
        :return: ``True`` if the action modified the grid, ``False`` for invalid or no-op actions
        :rtype: bool
        """
        if self.variant.cell_switching_mode:
            return self._apply_cell_switching_action(a)
        else:
            return self._apply_path_action(a)

    def _apply_cell_switching_action(self, a: int) -> bool:
        """Apply a cell-switching action that assigns or clears a non-endpoint cell.

        Decode ``a`` into a target cell and a color value where ``0`` means clear. Prevent changes to endpoint cells
        and update per-color connectivity flags for any affected colors.

        :param a: Encoded cell switching action index
        :type a: int
        :return: ``True`` when the action modified the grid, otherwise ``False`` for invalid actions
        :rtype: bool
        """
        r, c, color_value = self._decode_cell_switching_action(a)
        if r < 0:
            return False
        if self._endpoint_mask[r, c]:
            return False
        # Color value 0 means clear (code 0), values 1-N map to codes 1-N
        color_code: int = color_value

        affected: set[int] = set()
        if not self._bridges[r, c]:
            prev_color: int = self._grid_codes[r, c]
            if prev_color > 0:
                affected.add(prev_color - 1)
            self._grid_codes[r, c] = color_code
        else:
            prev_v: int = self._lane_v[r, c]
            prev_h: int = self._lane_h[r, c]
            if prev_v > 0:
                affected.add(prev_v - 1)
            if prev_h > 0:
                affected.add(prev_h - 1)
            self._lane_v[r, c] = color_code
            self._lane_h[r, c] = color_code

        if color_code > 0:
            affected.add(color_code - 1)
        self._refresh_cell_switch_connections(affected)
        return True

    def _refresh_cell_switch_connections(self, colors: Iterable[int]) -> None:
        """Recompute the :attr:`_closed` connection flags for a subset of colors.

        Only valid color indices within the range ``0..num_colors-1`` are considered. Use this method to avoid
        recomputing global connectivity after a local cell change.

        :param colors: Iterable of zero-based color indices to refresh
        :type colors: Iterable[int]
        """
        if not self.variant.cell_switching_mode:
            return

        unique: set[int] = {ci for ci in colors if 0 <= ci < self.num_colors}
        for ci in unique:
            self._closed[ci] = self._is_color_connected(ci)

    def _is_color_connected(self, ci: int) -> bool:
        """Return whether both endpoints of color ``ci`` are connected by its current path."""
        color_code: int = ci + 1
        start, goal = self._endpoints[ci]

        if start == goal:
            return True

        if not (self._cell_has_color(start, color_code) and self._cell_has_color(goal, color_code)):
            return False

        visited: NDArray[np.bool_] = self._visited_buffer
        visited.fill(False)
        queue: deque[Coord] = self._bfs_queue
        queue.clear()
        queue.append(start)
        visited[start[0], start[1]] = True

        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True

            for dr, dc in self._dirs:
                nr: int = r + dr
                nc: int = c + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    continue

                if visited[nr, nc]:
                    continue

                if self._cell_has_color((nr, nc), color_code):
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        return False

    def _cell_switch_connected(self, ci: int) -> bool:
        """Return whether a color's endpoints are connected in cell-switching mode."""
        return self._is_color_connected(ci)

    def _apply_path_action(self, a: int) -> bool:
        """Apply a path-building action for a color arm head.

        Decode the action to ``(color_index, head_index, direction)``. Valid moves include stepping into an empty
        cell, backtracking onto the previous cell of the same arm, joining the other head, or occupying an untouched
        endpoint. Bridge lane semantics are enforced when moving onto bridge cells.

        :param a: Encoded action index
        :type a: int
        :return: ``True`` if the action changed the board, otherwise ``False`` for invalid or blocked moves
        :rtype: bool
        """
        ci, hi, d = self._decode_action(a)
        if ci < 0:
            return False
        # Remove the check for _closed[ci] - allow actions even when connected
        head: Coord = self._heads[ci][hi]
        dr, dc = self._dirs[d]
        nr, nc = head[0] + dr, head[1] + dc
        if not (0 <= nr < self.H and 0 <= nc < self.W):
            return False

        tgt: Coord = (nr, nc)
        other_hi: int = 1 - hi
        other_head: Coord = self._heads[ci][other_hi]
        color_code: int = ci + 1
        stack: list[CellLane] = self._stacks[ci][hi]

        # Determine lane semantics for bridges
        move_lane: Lane = "n"
        # Lane only matters when stepping into a bridge cell
        if self._bridges[nr, nc]:
            move_lane = "v" if (d % 2 == 0) else "h"

        # Backtrack allowed (step onto previous cell for this arm)
        if len(stack) >= 2:
            prev_r, prev_c, _prev_lane = stack[-2]
            if (nr, nc) == (prev_r, prev_c):
                self._erase_last(ci, hi)
                # After backtracking, color may no longer be connected
                self._update_connection_status(ci)
                return True

        # Join other head → mark as connected (but don't lock!)
        if tgt == other_head:
            lane_here: Lane = "n" if not self._bridges[nr, nc] else ("v" if d % 2 == 0 else "h")
            # Mark occupancy to reflect the meeting cell (especially for bridges)
            self._occupy_cell(tgt, color_code, lane=lane_here)
            stack.append((nr, nc, lane_here))
            self._mark_presence(ci, hi, tgt)
            self._heads[ci][hi] = tgt
            self._closed[ci] = True
            return True

        # Join untouched other endpoint → mark as connected
        other_ep = self._endpoints[ci][other_hi]
        if tgt == other_ep and len(self._stacks[ci][other_hi]) == 1:
            stack.append((nr, nc, move_lane))
            self._occupy_cell(tgt, color_code, lane=move_lane)
            self._mark_presence(ci, hi, tgt)
            self._heads[ci][hi] = tgt
            self._closed[ci] = True
            return True

        # Occupancy checks
        if not self._can_occupy((nr, nc), color_code, lane=move_lane):
            return False

        # Avoid merging into the other arm interior (only head meeting is allowed)
        if self._arm_presence[ci, other_hi, nr, nc]:
            return False

        # Place segment
        self._occupy_cell((nr, nc), color_code, lane=move_lane)
        stack.append((nr, nc, move_lane))
        self._mark_presence(ci, hi, tgt)
        self._heads[ci][hi] = tgt

        # Update connection status - may have disconnected if was connected
        self._update_connection_status(ci)
        return True

    def _is_solved(self) -> bool:
        """Return whether the current environment state is a solved puzzle.

        In cell-switching mode this method delegates to :meth:`numberlink.env.NumberLinkRGBEnv._is_valid_solution`.
        In standard path-building mode all colors must be connected. When the active variant requires full coverage
        all cells must also be filled.

        :return: ``True`` if the puzzle is solved, otherwise ``False``
        :rtype: bool
        """
        if self.variant.cell_switching_mode:
            return self._is_valid_solution()

        if not np.all(self._closed):
            return False

        return not (self.variant.must_fill and not self._all_filled())

    @staticmethod
    def _is_deadlocked(action_mask: NDArray[np.uint8], solved: bool) -> bool:
        """Determine whether the environment is deadlocked.

        The environment is deadlocked when there are no available actions and the puzzle is not already solved.

        :param action_mask: Binary action mask where nonzero entries are valid
        :type action_mask: NDArray[np.uint8]
        :param solved: Whether the puzzle is already solved
        :type solved: bool
        :return: ``True`` when deadlocked, otherwise ``False``
        :rtype: bool
        """
        return not solved and np.sum(action_mask) == 0

    def _all_filled(self) -> bool:
        """Return whether all cells are occupied according to environment rules.

        For non-bridge cells the grid code must be nonzero. For bridge cells at least one lane must be occupied.

        :return: ``True`` when every cell is occupied, otherwise ``False``
        :rtype: bool
        """
        # Every cell must be occupied: normal cell has a code, bridge cell has at least one lane used
        normal_ok: np.bool_ = np.all((self._grid_codes != 0) | self._bridges)
        bridge_ok: np.bool_ = np.all((~self._bridges) | ((self._lane_v != 0) | (self._lane_h != 0)))
        return bool(normal_ok and bridge_ok)

    def _is_valid_solution(self) -> bool:
        """Return whether the current state is a valid NumberLink solution.

        Validation rules

        1. Endpoints of each color must be connected by a continuous path of that color
        2. Paths must not branch or contain loops
        3. If the active variant requires full coverage, every cell must be filled

        :return: ``True`` when the current state meets all validity rules, otherwise ``False``
        :rtype: bool
        """
        # Check if all cells are filled (if required)
        if self.variant.must_fill and not self._all_filled():
            return False

        # For each color, verify connectivity and path validity
        for ci in range(self.num_colors):
            ep0, ep1 = self._endpoints[ci]
            color_code: int = ci + 1

            visited_map: NDArray[np.bool_] = self._visited_buffer_alt
            visited_map.fill(False)
            queue: deque[Coord] = self._bfs_queue
            queue.clear()
            queue.append(ep0)
            visited_map[ep0[0], ep0[1]] = True
            found_other_endpoint: bool = ep0 == ep1

            while queue:
                r, c = queue.popleft()
                if (r, c) == ep1:
                    found_other_endpoint = True
                for dr, dc in self._dirs:
                    nr: int = r + dr
                    nc: int = c + dc
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    if visited_map[nr, nc]:
                        continue
                    if self._cell_has_color((nr, nc), color_code):
                        visited_map[nr, nc] = True
                        queue.append((nr, nc))

            if not found_other_endpoint:
                return False

            color_mask: NDArray[np.bool_] = self._visited_buffer
            color_mask.fill(False)
            non_bridge_mask: NDArray[np.bool_] = (~self._bridges) & (self._grid_codes == color_code)
            color_mask[non_bridge_mask] = True
            if np.any(self._bridges):
                bridge_mask: NDArray[np.bool_] = self._bridges
                v_match: NDArray[np.bool_] = (self._lane_v == color_code) & bridge_mask
                h_match: NDArray[np.bool_] = (self._lane_h == color_code) & bridge_mask
                color_mask[v_match | h_match] = True

            if np.any(color_mask & ~visited_map):
                return False

            coords: NDArray[np.intp] = np.argwhere(color_mask)
            for coord in coords:
                r = int(coord[0])
                c = int(coord[1])
                neighbor_count: int = 0
                for dr, dc in self._dirs:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    if color_mask[nr, nc]:
                        neighbor_count += 1

                cell_tuple: Coord = (r, c)
                if cell_tuple in {ep0, ep1}:
                    if neighbor_count != 1:
                        return False
                elif neighbor_count != 2:
                    return False

        return True

    # Occupancy helpers

    def _cell_has_color(self, cell: Coord, color_code: int) -> bool:
        """Return whether the given grid cell currently contains ``color_code``.

        For bridge cells check both vertical and horizontal lanes for the color.

        :param cell: Coordinate as ``(row, col)``
        :type cell: Coord
        :param color_code: One-based color code stored in the grid arrays
        :type color_code: int
        :return: ``True`` when the cell contains the color, otherwise ``False``
        :rtype: bool
        """
        r: int
        c: int
        r, c = cell
        if not self._bridges[r, c]:
            return self._grid_codes[r, c] == color_code

        return (self._lane_v[r, c] == color_code) or (self._lane_h[r, c] == color_code)

    def _can_occupy(self, cell: Coord, color_code: int, *, lane: Lane) -> bool:
        """Test whether a color can occupy a target cell for a specific lane.

        For non-bridge cells the cell must be empty. For bridge cells the requested lane must be empty or already
        occupied by the same color.

        :param cell: Target coordinate as ``(row, col)``
        :type cell: Coord
        :param color_code: One-based color code
        :type color_code: int
        :param lane: Lane specifier, one of ``'v'``, ``'h'``, or ``'n'`` for non-bridge
        :type lane: Lane
        :return: ``True`` if the occupancy is allowed, otherwise ``False``
        :rtype: bool
        """
        r, c = cell
        if not self._bridges[r, c]:
            return self._grid_codes[r, c] == 0
        if lane == "v":
            return self._lane_v[r, c] in {0, color_code}
        if lane == "h":
            return self._lane_h[r, c] in {0, color_code}
        return False

    def _occupy_cell(self, cell: Coord, color_code: int, *, lane: Lane) -> None:
        """Occupy a cell with the provided color code for the specified lane.

        For endpoints on bridge cells the ``lane`` argument may be ``'n'`` to indicate both lanes should be marked.

        :param cell: Target coordinate as ``(row, col)``
        :type cell: Coord
        :param color_code: One-based color code to write into the grid
        :type color_code: int
        :param lane: Lane specifier indicating which lane to set
        :type lane: Lane
        """
        r, c = cell
        if not self._bridges[r, c]:
            self._grid_codes[r, c] = color_code
            return
        if lane == "v":
            self._lane_v[r, c] = color_code
        elif lane == "h":
            self._lane_h[r, c] = color_code
        else:
            # endpoints on bridges: mark both lanes
            self._lane_v[r, c] = color_code
            self._lane_h[r, c] = color_code

    def _mark_presence(self, ci: int, hi: int, cell: Coord) -> None:
        """Mark a coordinate as occupied in the per-arm presence grid."""
        self._arm_presence[ci, hi, cell[0], cell[1]] = True

    def _clear_presence(self, ci: int, hi: int, cell: Coord) -> None:
        """Clear a coordinate from the per-arm presence grid."""
        self._arm_presence[ci, hi, cell[0], cell[1]] = False

    def _erase_last(self, ci: int, hi: int) -> None:
        """Erase the last placed segment for a color arm.

        Pop the last entry from the specified arm stack and clear the corresponding grid or lane value. Protect the
        starting endpoint for the specified head so it is not removed.

        :param ci: Zero-based color index
        :type ci: int
        :param hi: Head index for the arm, typically ``0`` or ``1``
        :type hi: int
        """
        stack: list[CellLane] = self._stacks[ci][hi]
        last_r: int
        last_c: int
        last_lane: Lane
        last_r, last_c, last_lane = stack.pop()

        # Only protect the starting endpoint for this head, not the other endpoint
        starting_endpoint: Coord = self._endpoints[ci][hi]
        last_cell: Coord = (last_r, last_c)
        if last_cell == starting_endpoint:
            stack.append((last_r, last_c, last_lane))
            return
        self._clear_presence(ci, hi, last_cell)

        if not self._bridges[last_r, last_c]:
            self._grid_codes[last_r, last_c] = 0
        elif last_lane == "v":
            self._lane_v[last_r, last_c] = 0
        elif last_lane == "h":
            self._lane_h[last_r, last_c] = 0
        self._heads[ci][hi] = (stack[-1][0], stack[-1][1])

    def clear_color_path(self, ci: int) -> None:
        """Clear all non-endpoint segments for color ``ci`` and reset its connection flag.

        This utility is used by the interactive viewer when switching the active drawing endpoint. Remove every
        non-endpoint segment for both arms while leaving the original endpoints intact.

        :param ci: Zero-based color index
        :type ci: int
        :raises ValueError: If ``ci`` is outside the valid range
        :raises RuntimeError: If invoked while the environment uses cell-switching mode
        """
        if not (0 <= ci < self.num_colors):
            raise ValueError(f"Color index {ci} out of range")
        if self.variant.cell_switching_mode:
            raise RuntimeError("clear_color_path is not supported in cell-switching mode")

        for hi in (0, 1):
            stack: list[CellLane] = self._stacks[ci][hi]
            while len(stack) > 1:
                self._erase_last(ci, hi)
            # Ensure heads precisely match the endpoints.
            self._heads[ci][hi] = self._endpoints[ci][hi]

        self._closed[ci] = False

    def _update_connection_status(self, ci: int) -> None:
        """Update the per-color connection flag by inspecting both arm heads.

        A color is connected when both heads share the same position or when one head has reached the opposite
        endpoint and that endpoint's stack length is greater than one.

        :param ci: Zero-based color index
        :type ci: int
        """
        self._closed[ci] = self._is_color_connected(ci)

    # Masks & rendering

    def _compute_action_mask(self) -> NDArray[np.uint8]:
        """Compute a binary action mask of valid actions for the current state.

        Delegate to :meth:`numberlink.env.NumberLinkRGBEnv._compute_cell_switching_mask` or
        :meth:`numberlink.env.NumberLinkRGBEnv._compute_path_mask` depending on the active variant.

        :return: One-dimensional array where nonzero entries mark valid actions
        :rtype: NDArray[np.uint8]
        """
        if self.variant.cell_switching_mode:
            return self._compute_cell_switching_mask()

        return self._compute_path_mask()

    def _compute_cell_switching_mask(self) -> NDArray[np.uint8]:
        """Compute the action mask for cell-switching mode.

        If :attr:`_cell_switch_mask` is ``None`` return a zero mask covering the full action space. Otherwise return a
        copy of the configured mask.

        :return: Action mask for the cell-switching action space
        :rtype: NDArray[np.uint8]
        """
        if self._cell_switch_mask is None:
            colors_plus_clear: int = self.num_colors + 1
            total_actions: int = self.H * self.W * colors_plus_clear
            return np.zeros((total_actions,), dtype=np.uint8)

        return self._cell_switch_mask.copy()

    def _compute_path_mask(self) -> NDArray[np.uint8]:
        """Compute the action mask for standard path-building mode.

        The mask marks legal head moves for every color and arm head. Legal moves include backtracking, meeting the
        other head, stepping onto an untouched endpoint, and moving into an empty lane or cell while respecting
        bridge semantics.

        :return: One-dimensional binary mask for the discrete action space
        :rtype: NDArray[np.uint8]
        """
        mask: NDArray[np.uint8] = self._path_action_mask
        if mask.size == 0:
            return mask
        mask.fill(0)
        num_dirs: int = self._num_dirs
        dirs: NDArray[np.int8] = self._dirs
        bridges: NDArray[np.bool_] = self._bridges
        heads: list[list[Coord]] = self._heads
        stacks: list[list[list[CellLane]]] = self._stacks
        endpoints: list[tuple[Coord, Coord]] = self._endpoints
        presence: NDArray[np.bool_] = self._arm_presence
        actions_per_color: int = self._actions_per_color
        for ci in range(self.num_colors):
            heads_ci: list[Coord] = heads[ci]
            stacks_ci: list[list[CellLane]] = stacks[ci]
            endpoints_ci: tuple[Coord, Coord] = endpoints[ci]
            presence_ci: NDArray[np.bool_] = presence[ci]
            color_code: int = ci + 1
            base_color_index: int = ci * actions_per_color
            for hi in (0, 1):
                head_r, head_c = heads_ci[hi]
                arm_stack: list[CellLane] = stacks_ci[hi]
                stack_len: int = len(arm_stack)
                prev_r: int | None = None
                prev_c: int | None = None
                if stack_len >= 2:
                    prev_r, prev_c, _prev_lane = arm_stack[-2]
                other_hi: int = 1 - hi
                other_head_r, other_head_c = heads_ci[other_hi]
                other_endpoint: Coord = endpoints_ci[other_hi]
                other_stack_len: int = len(stacks_ci[other_hi])
                presence_other: NDArray[np.bool_] = presence_ci[other_hi]
                base_index: int = base_color_index + hi * num_dirs
                for d in range(num_dirs):
                    dr_i: int = int(dirs[d][0])
                    dc_i: int = int(dirs[d][1])
                    nr: int = head_r + dr_i
                    nc: int = head_c + dc_i
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    action_index: int = base_index + d
                    if prev_r is not None and nr == prev_r and nc == prev_c:
                        mask[action_index] = 1
                        continue
                    if nr == other_head_r and nc == other_head_c:
                        mask[action_index] = 1
                        continue
                    if nr == other_endpoint[0] and nc == other_endpoint[1] and other_stack_len == 1:
                        mask[action_index] = 1
                        continue
                    move_lane: Lane = "n"
                    if bridges[nr, nc]:
                        move_lane = "v" if (d % 2 == 0) else "h"
                    if not self._can_occupy((nr, nc), color_code, lane=move_lane):
                        continue
                    if presence_other[nr, nc]:
                        continue
                    mask[action_index] = 1

        return mask.copy()

    def _render_rgb(self) -> ObsType:
        """Produce an RGB image representing the current grid state.

        Normal cells are filled from the palette based on :attr:`_grid_codes`. Bridge cells are composed by mixing
        lane colors when both lanes are present. Apply endpoint styling according to the active render configuration.

        :return: RGB image with shape ``(H, W, 3)`` and dtype ``uint8``
        :rtype: ObsType
        """
        img: ObsType = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        bg: NDArray[np.uint8] = self._grid_background_color
        if np.any(bg != 0):
            img[:] = bg

        # Fill normal cells directly from palette lookup
        normal_mask: NDArray[np.bool_] = (~self._bridges) & (self._grid_codes > 0)
        if np.any(normal_mask):
            color_indices: NDArray[np.unsignedinteger] = self._grid_codes[normal_mask] - 1
            img[normal_mask] = self._palette_stack[color_indices]

        # Populate bridge cells by mixing lane content
        if self._bridge_rows.size > 0:
            rows = self._bridge_rows
            cols = self._bridge_cols
            v_codes: NDArray[np.unsignedinteger] = self._lane_v[rows, cols]
            h_codes: NDArray[np.unsignedinteger] = self._lane_h[rows, cols]
            bridge_colors: NDArray[np.uint8]
            if self._bridge_color_buffer is None:
                bridge_colors = np.zeros((rows.size, 3), dtype=np.uint8)
            else:
                bridge_colors = self._bridge_color_buffer
                bridge_colors.fill(0)

            only_v: NDArray[np.bool_] = (v_codes != 0) & (h_codes == 0)
            if np.any(only_v):
                bridge_colors[only_v] = self._palette_stack[v_codes[only_v] - 1]

            only_h: NDArray[np.bool_] = (h_codes != 0) & (v_codes == 0)
            if np.any(only_h):
                bridge_colors[only_h] = self._palette_stack[h_codes[only_h] - 1]

            both: NDArray[np.bool_] = (v_codes != 0) & (h_codes != 0)
            if np.any(both):
                mixed: NDArray[np.int16] = (
                    self._palette_stack[v_codes[both] - 1].astype(np.int16)
                    + self._palette_stack[h_codes[both] - 1].astype(np.int16)
                ) // 2
                bridge_colors[both] = mixed.astype(np.uint8)

            img[rows, cols] = bridge_colors

        # Apply endpoint base color or adjusted color based on render config
        ep_rows: NDArray[np.intp] | None = None
        ep_cols: NDArray[np.intp] | None = None
        if self._endpoints_array.size > 0:
            ep_rows = self._endpoints_array[:, :, 0].reshape(-1).astype(np.intp, copy=False)
            ep_cols = self._endpoints_array[:, :, 1].reshape(-1).astype(np.intp, copy=False)

            if self._render_cfg.endpoint_border_thickness > 0:
                img[ep_rows, ep_cols] = self._endpoint_base_colors
            elif self._endpoint_adjusted_colors is not None:
                img[ep_rows, ep_cols] = self._endpoint_adjusted_colors

        # Upscale if needed
        if self._pixels_per_cell_h > 1 or self._pixels_per_cell_w > 1:
            img = np.repeat(img, self._pixels_per_cell_h, axis=0)
            img = np.repeat(img, self._pixels_per_cell_w, axis=1)

        # Draw endpoint borders after gridlines so borders are visible (only when enough pixels per cell)
        if (
            self._render_cfg.endpoint_border_thickness > 0
            and self._endpoints_array.size > 0
            and self._pixels_per_cell_h >= 2
            and self._pixels_per_cell_w >= 2
        ):
            border_color: NDArray[np.uint8] = np.array(self._render_cfg.endpoint_border_color, dtype=np.uint8)
            ep_border_thickness: int = self._render_cfg.endpoint_border_thickness
            grid_thickness_for_inset: int = (
                max(1, int(self._render_cfg.gridline_thickness)) if self._render_cfg.gridline_color is not None else 0
            )
            ep_rows_loc: NDArray[np.intp] = self._endpoints_array[:, :, 0].reshape(-1).astype(np.intp, copy=False)
            ep_cols_loc: NDArray[np.intp] = self._endpoints_array[:, :, 1].reshape(-1).astype(np.intp, copy=False)

            for ep_r, ep_c in zip(ep_rows_loc, ep_cols_loc, strict=True):
                r0: int = ep_r * self._pixels_per_cell_h
                r1: int = r0 + self._pixels_per_cell_h
                c0: int = ep_c * self._pixels_per_cell_w
                c1: int = c0 + self._pixels_per_cell_w

                if grid_thickness_for_inset > 0:
                    hpx, wpx = img.shape[0], img.shape[1]

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
                interior_h: int = max(0, ir1 - ir0)
                interior_w: int = max(0, ic1 - ic0)
                if interior_h <= 0 or interior_w <= 0:
                    continue

                t: int = min(ep_border_thickness, interior_h, interior_w)
                if t <= 0:
                    continue

                # Top border (inner)
                img[ir0 : ir0 + t, ic0:ic1] = border_color
                # Bottom border (inner)
                img[ir1 - t : ir1, ic0:ic1] = border_color
                # Left border (inner)
                img[ir0:ir1, ic0 : ic0 + t] = border_color
                # Right border (inner)
                img[ir0:ir1, ic1 - t : ic1] = border_color

        # Render endpoint numbers last so they are not occluded by gridlines or borders
        if self._render_cfg.show_endpoint_numbers and self._pixels_per_cell_h >= 10 and self._pixels_per_cell_w >= 10:
            specs: list[tuple[str, Coord, RGBInt, int]] = build_endpoint_labels(
                endpoints=self._endpoints,
                pixels_per_cell_h=self._pixels_per_cell_h,
                pixels_per_cell_w=self._pixels_per_cell_w,
                min_scale=self._render_cfg.number_font_min_scale,
                max_scale=self._render_cfg.number_font_max_scale,
                gridline_thickness=self._render_cfg.gridline_thickness or 0,
            )
            num_color: RGBInt = (
                self._render_cfg.number_font_color[0],
                self._render_cfg.number_font_color[1],
                self._render_cfg.number_font_color[2],
            )
            num_border: RGBInt = (
                self._render_cfg.number_font_border_color[0],
                self._render_cfg.number_font_border_color[1],
                self._render_cfg.number_font_border_color[2],
            )
            for text, center, _color, scale in specs:
                render_bitmap_text_centered(
                    text, center, num_color, img, scale, num_border, self._render_cfg.number_font_border_thickness
                )

        # Draw gridlines last so they appear on top of numbers, endpoint borders, and other content
        if self._render_cfg.gridline_color is not None and (
            self._pixels_per_cell_h >= 2 and self._pixels_per_cell_w >= 2
        ):
            gl_color: NDArray[np.uint8] = np.array(self._render_cfg.gridline_color, dtype=np.uint8)
            grid_thickness: int = max(1, self._render_cfg.gridline_thickness)
            hpx, wpx = img.shape[0], img.shape[1]
            cell_h, cell_w = self._pixels_per_cell_h, self._pixels_per_cell_w

            # Vertical lines
            for c in range(self.W + 1):
                x: int = c * cell_w
                x0: int = max(0, x - grid_thickness // 2)
                x1: int = min(wpx, x0 + grid_thickness)
                img[:, x0:x1] = gl_color

            # Horizontal lines
            for r in range(self.H + 1):
                y: int = r * cell_h
                y0: int = max(0, y - grid_thickness // 2)
                y1: int = min(hpx, y0 + grid_thickness)
                img[y0:y1, :] = gl_color

        return img

    def _render_text(self) -> str:
        """Produce a compact textual representation of the grid for terminal output.

        The header includes step count and the number of connected colors. The grid uses these symbols

        - ``.`` for empty regular cells
        - Uppercase letter for endpoints
        - Lowercase letter for path segments
        - For bridge cells the first character represents the vertical lane and the second the horizontal lane. Use
            the same casing rules, or ``*`` when the lane is empty

        :return: Multi-line string representing the grid and status line
        :rtype: str
        """
        rev_map: dict[int, str] = {i + 1: self._letters[i] for i in range(self.num_colors)}
        rows: list[str] = []

        # Precompute reusable state
        action_mask: NDArray[np.uint8] = self._compute_action_mask()
        solved: bool = self._is_solved()
        deadlocked: bool = self._is_deadlocked(action_mask, solved)
        truncated: bool = self._steps >= self.max_steps and not solved and not deadlocked

        endpoint_lookup: dict[Coord, int] = {}
        for ci, pair in enumerate(self._endpoints, start=1):
            endpoint_lookup[pair[0]] = ci
            endpoint_lookup[pair[1]] = ci

        # Add header with status
        status_parts: list[str] = [f"Steps: {self._steps}"]
        connected_count: int = int(np.sum(self._closed))
        status_parts.append(f"Connected: {connected_count}/{self.num_colors}")

        # Check terminal and truncation status
        if solved:
            status_parts.append("Solved")
        elif deadlocked:
            status_parts.append("Deadlocked")
        elif truncated:
            status_parts.append("Truncated")

        rows.append(" | ".join(status_parts))
        rows.append("-" * max(self.W * 2, len(rows[0])))

        # Render grid
        for r in range(self.H):
            line: list[str] = []
            for c in range(self.W):
                if self._bridges[r, c]:
                    v, h = self._lane_v[r, c], self._lane_h[r, c]
                    v_char: str
                    h_char: str
                    if v != 0:
                        letter_v: str = rev_map[int(v)]
                        v_char = letter_v if endpoint_lookup.get((r, c)) == int(v) else letter_v.lower()
                    else:
                        v_char = "*"

                    if h != 0:
                        letter_h: str = rev_map[int(h)]
                        h_char = letter_h if endpoint_lookup.get((r, c)) == int(h) else letter_h.lower()
                    else:
                        h_char = "*"

                    line.append(f"{v_char}{h_char}")
                    continue

                code: int = self._grid_codes[r, c]
                if code == 0:
                    line.append(". ")
                else:
                    letter: str = rev_map[code]
                    if endpoint_lookup.get((r, c)) == code:
                        line.append(f"{letter} ")
                    else:
                        line.append(f"{letter.lower()} ")

            rows.append("".join(line))
        return "\n".join(rows)
