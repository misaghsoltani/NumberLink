"""Shared level setup utilities for NumberLink environments.

This module provides helpers to resolve level sources, validate grids, build color palettes, and construct a
:class:`LevelTemplate` used by the environment creation code. Use :func:`build_level_template` as the entry point to
produce reusable static data for a level before creating an environment with the :mod:`numberlink` package.

.. note::
    Type information is provided in function signatures where available and is compatible with static analysis tools.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .config import RenderConfig, RewardConfig, VariantConfig
from .generator import generate_level
from .levels import LEVELS, Level

if TYPE_CHECKING:
    from .config import GeneratorConfig
    from .types import Coord, RGBInt


@dataclass(frozen=True, slots=True)
class LevelTemplate:
    """Precomputed static data for a NumberLink level configuration.

    Instances of this dataclass hold static, precomputed values required to create runtime environments.
    Fields are documented in the class body via type annotations. Construct instances using :func:`build_level_template`
    to ensure internal consistency.
    """

    variant: VariantConfig
    reward_config: RewardConfig
    render_config: RenderConfig
    raw_grid: list[str]
    bridges_mask: NDArray[np.bool_]
    level_id: str | None
    letters: list[str]
    palette_map: dict[str, RGBInt]
    palette_arrays: list[NDArray[np.uint8]]
    endpoints: NDArray[np.int16]  # (num_colors, 2, 2)
    dirs: NDArray[np.int8]
    num_dirs: int
    actions_per_color: int
    action_space_size: int
    cell_switch_action_space_size: int
    height: int
    width: int
    num_colors: int
    solution: list[list[Coord]] | None


def resolve_variant(variant: VariantConfig | None, generator: GeneratorConfig | None) -> VariantConfig:
    """Resolve the effective :class:`VariantConfig` to use for level construction.

    If ``variant`` is provided, return it unchanged. If ``variant`` is ``None`` and ``generator`` is provided, derive
    values from the generator configuration while preserving explicit defaults from :class:`VariantConfig` where
    applicable. The returned object is safe to use with :func:`build_level_template` and :func:`_resolve_level_source`.

    :param variant: Explicit variant configuration or ``None`` to derive one.
    :type variant: VariantConfig or None
    :param generator: Generator configuration used to derive variant fields when ``variant`` is ``None``.
    :type generator: GeneratorConfig or None

    :return: Effective variant configuration.
    :rtype: VariantConfig
    """
    if variant is not None:
        return variant

    if generator is None:
        return VariantConfig()

    default_variant = VariantConfig()
    bridges_enabled_default: bool = default_variant.bridges_enabled
    return VariantConfig(
        must_fill=True,
        allow_diagonal=False,
        bridges_enabled=generator.bridges_probability > 0.0 or bridges_enabled_default,
        cell_switching_mode=default_variant.cell_switching_mode,
    )


def _metric(a: Coord, b: Coord, *, allow_diagonal: bool) -> int:
    """Compute grid distance between two coordinates.

    Returns the Chebyshev distance when ``allow_diagonal`` is ``True`` and the Manhattan distance otherwise.
    This helper is internal and is used by :func:`_grid_ok` for validating path lengths between endpoints.

    :param a: First coordinate given as a pair ``(row, col)``.
    :type a: tuple[int, int]
    :param b: Second coordinate given as a pair ``(row, col)``.
    :type b: tuple[int, int]
    :param allow_diagonal: Whether diagonal moves are allowed.
    :type allow_diagonal: bool

    :return: Integer distance between the two coordinates.
    :rtype: int
    """
    dr: int = abs(a[0] - b[0])
    dc: int = abs(a[1] - b[1])
    return max(dr, dc) if allow_diagonal else dr + dc


def _grid_ok(grid: Sequence[str], *, min_path_length: int, allow_diagonal: bool) -> bool:
    """Validate that a grid contains valid endpoint pairs and path lengths.

    The function checks that every non-``'.'`` character appears exactly twice. For each pair it verifies that the
    distance computed by :func:`_metric` meets or exceeds ``min_path_length``. This routine is used by
    :func:`_resolve_level_source` when regenerating candidate levels from a :class:`GeneratorConfig`.

    :param grid: Iterable of row strings representing the grid.
    :type grid: Sequence[str]
    :param min_path_length: Minimum allowed distance between matching endpoints.
    :type min_path_length: int
    :param allow_diagonal: Whether diagonal moves are counted by :func:`_metric`.
    :type allow_diagonal: bool

    :return: ``True`` when the grid is valid, otherwise ``False``.
    :rtype: bool
    """
    letter_pos: dict[str, list[Coord]] = {}
    pts: list[Coord]
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == ".":
                continue

            pts = letter_pos.setdefault(ch, [])
            if len(pts) == 2:  # Fast-fail: more than two occurrences
                return False

            pts.append((r, c))

    for pts in letter_pos.values():
        if len(pts) != 2 or _metric(pts[0], pts[1], allow_diagonal=allow_diagonal) < min_path_length:
            return False

    return True


def _resolve_level_source(
    *,
    grid: Sequence[str] | None,
    level_id: str | None,
    variant: VariantConfig,
    bridges: Iterable[Coord] | None,
    generator: GeneratorConfig | None,
) -> tuple[list[str], set[Coord], str | None, list[list[Coord]] | None]:
    """Determine the resolved grid, bridge coordinates, level identifier, and solution.

    The resolution order is as follows:

    1. If ``generator`` is provided, attempt to generate candidate levels using :func:`generate_level` until a candidate
       satisfies :func:`_grid_ok` or a maximum number of attempts is reached.
    2. If ``level_id`` is provided, load the level from the :data:`LEVELS` mapping.
    3. If ``grid`` is provided, use it directly.
    4. Fall back to the built-in level with id ``'builtin_5x5_rw_4c'``.

    Caller-specified ``bridges`` override any bridges from generated or lookup sources. This helper is private and
    intended for use by :func:`build_level_template`.

    :param grid: Optional explicit grid to use.
    :type grid: Sequence[str] or None
    :param level_id: Optional level identifier to load from :data:`LEVELS`.
    :type level_id: str or None
    :param variant: Effective :class:`VariantConfig` used when validating generated candidates.
    :type variant: VariantConfig
    :param bridges: Optional iterable of bridge coordinates to force.
    :type bridges: Iterable[Coord] or None
    :param generator: Optional :class:`GeneratorConfig` used to generate candidate grids.
    :type generator: GeneratorConfig or None

    :return: Tuple ``(resolved_grid, resolved_bridges, resolved_level_id, resolved_solution)`` where ``resolved_grid``
        is a list of row strings, ``resolved_bridges`` is a set of coordinate tuples, ``resolved_level_id`` is either
        the selected level id or ``None`` when the grid came from a generator or explicit ``grid`` argument, and
        ``resolved_solution`` is the solution paths or ``None`` if unavailable.
    :rtype: tuple[list[str], set[Coord], str or None, list[list[Coord]] or None]
    """
    resolved_grid: list[str]
    resolved_bridges: set[Coord]
    resolved_level_id: str | None
    resolved_solution: list[list[Coord]] | None
    lvl: Level
    last_lvl: Level | None

    if generator is not None:
        cfg: GeneratorConfig = generator
        last_lvl = None

        for _ in range(cfg.max_retries):
            lvl = generate_level(cfg, variant=variant)
            last_lvl = lvl

            if _grid_ok(lvl.grid, min_path_length=generator.min_path_length, allow_diagonal=variant.allow_diagonal):
                resolved_grid = lvl.grid
                resolved_bridges = set(lvl.bridges or ())
                resolved_level_id = None
                resolved_solution = lvl.solution
                break

            if cfg.seed is not None:
                cfg = replace(cfg, seed=int(cfg.seed) + 1)

        else:
            # True "last candidate" fallback without regenerating
            lvl = last_lvl or generate_level(cfg, variant=variant)
            resolved_grid = lvl.grid
            resolved_bridges = set(lvl.bridges or ())
            resolved_level_id = None
            resolved_solution = lvl.solution

    elif level_id is not None:
        if level_id not in LEVELS:
            raise KeyError(f"Unknown level_id: {level_id}")

        lvl = LEVELS[level_id]
        resolved_grid = lvl.grid
        resolved_bridges = set(lvl.bridges or ())
        resolved_level_id = level_id
        resolved_solution = lvl.solution

    elif grid is not None:
        resolved_grid = list(grid)
        resolved_bridges = set(bridges or ())
        resolved_level_id = None
        resolved_solution = None

    else:
        lvl = LEVELS["builtin_5x5_rw_4c"]
        resolved_grid = lvl.grid
        resolved_bridges = set(lvl.bridges or ())
        resolved_level_id = "builtin_5x5_rw_4c"
        resolved_solution = lvl.solution

    # Caller-specified bridges always win
    if bridges is not None:
        resolved_bridges = set(bridges)

    return resolved_grid, resolved_bridges, resolved_level_id, resolved_solution


def _validate_grid(grid: Sequence[str]) -> Coord:
    """Validate grid shape and return its height and width.

    The function ensures the grid has at least one row and that all rows are the same width. It returns a tuple
    ``(height, width)`` for convenience. This helper is used by :func:`build_level_template` to validate input grids.

    :param grid: Sequence of row strings representing the grid.
    :type grid: Sequence[str]

    :return: Pair of integers ``(height, width)`` describing the grid size.
    :rtype: tuple[int, int]

    :raises ValueError: If the grid is empty or rows have inconsistent lengths.
    """
    height: int = len(grid)
    if height <= 0:
        raise ValueError("Grid must have at least one row")

    width: int = len(grid[0])
    for row in grid:
        if len(row) != width:
            raise ValueError("Grid must be rectangular")

    return height, width


def _generate_colors(num_colors: int) -> list[RGBInt]:
    """Generate visually distinguishable RGB colors.

    The function returns a list of RGB triples where each triple is an ``(R, G, B)`` tuple with values in the range
    ``0`` to ``255``. For small values of ``num_colors`` a curated palette of high-contrast colors is returned.
    For larger values additional colors are generated by sampling HSV space and converting values to RGB.

    :param num_colors: Number of distinct colors required.
    :type num_colors: int

    :return: List of RGB color tuples suitable for use in rendering.
    :rtype: list[RGBInt]
    """
    # Distinguishable colors (based on color theory and perceptual studies)
    base_palette: list[RGBInt] = [
        (230, 25, 75),  # Red
        (60, 180, 75),  # Green
        (0, 130, 200),  # Blue
        (245, 130, 48),  # Orange
        (145, 30, 180),  # Purple
        (70, 240, 240),  # Cyan
        (240, 50, 230),  # Magenta
        (250, 190, 212),  # Pink
        (0, 128, 128),  # Teal
        (220, 190, 255),  # Lavender
        (170, 110, 40),  # Brown
        (255, 250, 200),  # Beige
        (128, 0, 0),  # Maroon
        (170, 255, 195),  # Mint
        (128, 128, 0),  # Olive
        (255, 215, 180),  # Coral
        (0, 0, 128),  # Navy
        (128, 128, 128),  # Grey
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Aqua
    ]

    if num_colors <= len(base_palette):
        return base_palette[:num_colors]

    # For more colors, generate additional ones using HSV space
    # Start with the base palette and add evenly distributed hues
    result: list[RGBInt] = base_palette.copy()

    additional_needed: int = num_colors - len(base_palette)
    for i in range(additional_needed):
        # Generate colors with varying hue but high saturation and value
        hue: float = (i * 360.0 / additional_needed + 180.0) % 360.0  # Offset to avoid base colors
        saturation: float = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
        value: float = 0.8 + (i % 2) * 0.15  # Vary brightness slightly

        # Convert HSV to RGB
        h_sector: int = int(hue / 60.0)
        f: float = hue / 60.0 - h_sector
        p: float = value * (1.0 - saturation)
        q: float = value * (1.0 - f * saturation)
        t: float = value * (1.0 - (1.0 - f) * saturation)

        r: float
        g: float
        b: float

        if h_sector == 0:
            r, g, b = value, t, p
        elif h_sector == 1:
            r, g, b = q, value, p
        elif h_sector == 2:
            r, g, b = p, value, t
        elif h_sector == 3:
            r, g, b = p, q, value
        elif h_sector == 4:
            r, g, b = t, p, value
        else:
            r, g, b = value, p, q

        result.append((int(r * 255), int(g * 255), int(b * 255)))

    return result


def _build_palette(
    letters: list[str], palette: dict[str, RGBInt] | None
) -> tuple[dict[str, RGBInt], list[NDArray[np.uint8]]]:
    """Build a mapping from letter identifiers to RGB colors and return numpy arrays for rendering.

    The function first applies any user-provided mappings from ``palette``. For letters that remain unassigned it
    generates visually distinguishable colors using :func:`_generate_colors`. The function also returns
    a list of ``numpy`` arrays ordered to match ``letters`` which is convenient for rendering code.

    :param letters: Ordered list of letter identifiers present in the grid.
    :type letters: list[str]
    :param palette: Optional mapping of letters to RGB triples to use as a base palette.
    :type palette: dict[str, RGBInt] or None

    :return: Tuple of ``(palette_map, palette_arrays)`` where ``palette_map`` is a mapping from letter to RGB tuple and
        ``palette_arrays`` is a list of ``numpy.uint8`` arrays in the same order as ``letters``.
    :rtype: tuple[dict[str, RGBInt], list[NDArray[numpy.uint8]]]
    """
    palette_map: dict[str, RGBInt] = {}

    # First, use any user-provided custom colors
    if palette:
        palette_map.update(palette)

    # For any letters without assigned colors, generate distinguishable colors
    unassigned_letters: list[str] = [letter for letter in letters if letter not in palette_map]

    if unassigned_letters:
        # Generate enough distinguishable colors for all unassigned letters
        num_needed: int = len(unassigned_letters)
        distinguishable_colors: list[RGBInt] = _generate_colors(num_needed)

        # Assign generated colors to unassigned letters
        for letter, color in zip(unassigned_letters, distinguishable_colors, strict=True):
            palette_map[letter] = color

    palette_arrays: list[NDArray[np.uint8]] = [np.array(palette_map[letter], dtype=np.uint8) for letter in letters]
    return palette_map, palette_arrays


def build_level_template(
    *,
    grid: Sequence[str] | None,
    level_id: str | None,
    variant: VariantConfig | None,
    bridges: Iterable[Coord] | None,
    generator: GeneratorConfig | None,
    reward_config: RewardConfig | None,
    render_config: RenderConfig | None,
    palette: dict[str, RGBInt] | None,
    solution: list[list[Coord]] | None = None,
) -> LevelTemplate:
    """Construct a :class:`LevelTemplate` from configuration inputs.

    This function resolves the effective variant using :func:`resolve_variant` and chooses a grid using
    :func:`_resolve_level_source`. It validates the resolved grid with :func:`_validate_grid`, builds a color palette
    with :func:`_build_palette`, and prepares arrays and masks required by runtime environments.

    The returned :class:`LevelTemplate` contains precomputed values such as endpoint coordinates, direction vectors,
    action space sizes, and color palettes. Use the returned object to construct environment instances.

    :param grid: Optional explicit grid to use. If ``None`` and ``generator`` is provided then levels are generated
        until a valid candidate is found.
    :type grid: Sequence[str] or None
    :param level_id: Optional key into the :data:`LEVELS` mapping to select a predefined level.
    :type level_id: str or None
    :param variant: Optional :class:`VariantConfig` to apply. If ``None`` the value is derived using
        :func:`resolve_variant` when ``generator`` is present.
    :type variant: VariantConfig or None
    :param bridges: Optional iterable of bridge coordinates to force. If provided these override any bridges from
        generated or predefined levels.
    :type bridges: Iterable[Coord] or None
    :param generator: Optional :class:`GeneratorConfig` used to generate levels when a grid is not supplied.
    :type generator: GeneratorConfig or None
    :param reward_config: Optional :class:`RewardConfig` to attach to the template. When ``None`` a default
        :class:`RewardConfig` is used.
    :type reward_config: RewardConfig or None
    :param render_config: Optional :class:`RenderConfig` that controls rendering parameters. When ``None`` a default
        :class:`RenderConfig` is used.
    :type render_config: RenderConfig or None
    :param palette: Optional mapping from letters to RGB triples. Missing letters are assigned generated colors by
        :func:`_build_palette`.
    :type palette: dict[str, RGBInt] or None

    :return: A fully populated :class:`LevelTemplate` suitable for environment construction.
    :rtype: LevelTemplate

    :raises ValueError: If the resolved grid contains fewer than one color pair or a letter doesn't appear twice.
    """
    effective_variant: VariantConfig = resolve_variant(variant, generator)
    effective_reward: RewardConfig = reward_config if reward_config is not None else RewardConfig()
    effective_render: RenderConfig = render_config if render_config is not None else RenderConfig()

    resolved_grid: list[str]
    resolved_bridges: Iterable[Coord]
    resolved_level_id: str | None
    resolved_solution: list[list[Coord]] | None
    resolved_grid, resolved_bridges, resolved_level_id, resolved_solution = _resolve_level_source(
        grid=grid, level_id=level_id, variant=effective_variant, bridges=bridges, generator=generator
    )

    if solution is not None:
        resolved_solution = solution

    height: int
    width: int
    height, width = _validate_grid(resolved_grid)

    bridge_mask: NDArray[np.bool_] = np.zeros((height, width), dtype=np.bool_)
    if effective_variant.bridges_enabled:
        for r, c in resolved_bridges:
            if 0 <= r < height and 0 <= c < width:
                bridge_mask[r, c] = True

    else:
        bridge_mask.fill(False)

    letters: list[str] = sorted({ch for row in resolved_grid for ch in row if ch != "."})
    if len(letters) < 1:
        raise ValueError("Grid must contain at least one color pair")

    palette_map: dict[str, RGBInt]
    palette_arrays: list[NDArray[np.uint8]]
    palette_map, palette_arrays = _build_palette(letters, palette)

    endpoints: list[list[Coord]] = []
    for letter in letters:
        coords: list[Coord] = [
            (r, c) for r, row in enumerate(resolved_grid) for c, ch in enumerate(row) if ch == letter
        ]
        if len(coords) != 2:
            raise ValueError(f"Color '{letter}' must appear exactly twice (found {len(coords)})")

        endpoints.append(coords)

    endpoint_array: NDArray[np.int16] = np.array(endpoints, dtype=np.int16)

    dirs: NDArray[np.int8] = (
        np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]], dtype=np.int8)
        if effective_variant.allow_diagonal
        else np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int8)
    )
    num_dirs: int = dirs.shape[0]
    actions_per_color: int = 2 * num_dirs

    num_colors: int = len(letters)
    total_actions: int = num_colors * actions_per_color
    cell_switch_actions: int = height * width * (num_colors + 1)

    return LevelTemplate(
        variant=effective_variant,
        reward_config=effective_reward,
        render_config=effective_render,
        raw_grid=resolved_grid,
        bridges_mask=bridge_mask,
        level_id=resolved_level_id,
        letters=letters,
        palette_map=palette_map,
        palette_arrays=palette_arrays,
        endpoints=endpoint_array,
        dirs=dirs,
        num_dirs=num_dirs,
        actions_per_color=actions_per_color,
        action_space_size=total_actions,
        cell_switch_action_space_size=cell_switch_actions,
        height=height,
        width=width,
        num_colors=num_colors,
        solution=resolved_solution,
    )
