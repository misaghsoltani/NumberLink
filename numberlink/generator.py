"""Procedural level generator for the :mod:`numberlink` environment.

This module implements two procedural generators and a set of internal helpers used to create valid NumberLink puzzles.
The primary entry point is :func:`generate_level`. The function :func:`generate_level` dispatches to
:func:`_gen_hamiltonian_partition` or :func:`_gen_random_walk` depending on the provided configuration. All generator
functions return a :class:`numberlink.levels.Level` representing the puzzle grid and optional bridge positions.

The implementation organizes helpers for path creation, rewiring, partitioning, validation, and rendering.
See individual function docstrings for algorithmic notes and usage details.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .levels import Level

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.random import Generator
    from numpy.typing import NDArray

    from .config import GeneratorConfig, VariantConfig
    from .types import Coord

DIR_STEPS: tuple[Coord, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


# Generator helpers


def generate_level(cfg: GeneratorConfig, *, variant: VariantConfig | None = None) -> Level:
    """Generate a puzzle level using the configured generator settings.

    Dispatch to either :func:`_gen_hamiltonian_partition` or :func:`_gen_random_walk` depending on the value of
    ``cfg.mode``. The returned value is a :class:`numberlink.levels.Level` instance suitable for immediate use by the
    environment or saving to disk.

    :param cfg: Generator configuration object. See :class:`numberlink.config.GeneratorConfig` for options and defaults.
    :type cfg: GeneratorConfig
    :param variant: Variant configuration object. If ``None``, defaults to ``VariantConfig()`` with standard settings.
        The ``must_fill`` and ``allow_diagonal`` settings are taken from this parameter.
    :type variant: VariantConfig or None

    :return: Generated puzzle level as a :class:`numberlink.levels.Level`.
    :rtype: Level

    :raises ValueError: If ``cfg.mode`` is not one of ``'hamiltonian'`` or ``'random_walk'``.
    """
    if variant is None:
        variant = VariantConfig()

    if cfg.mode == "hamiltonian":
        return _gen_hamiltonian_partition(cfg, variant=variant)
    elif cfg.mode == "random_walk":
        return _gen_random_walk(cfg, variant=variant)
    else:
        raise ValueError(f"Unknown generator mode: {cfg.mode}. Must be 'hamiltonian' or 'random_walk'.")


def _gen_hamiltonian_partition(
    cfg: GeneratorConfig, *, variant: VariantConfig, allow_random_fallback: bool = True
) -> Level:
    """Deterministic generator that assembles a valid, fully filled puzzle.

    The generator builds a Hamiltonian path that covers every cell and partitions that path into segments that become
    colored paths for the puzzle. The high level steps are:

    1. Build a Hamiltonian backbone using a serpentine sweep implemented by :func:`_build_initial_path`.
    2. Apply randomized 2 opt style rewiring to the backbone using :func:`_apply_loop_moves` to introduce variety while
       preserving full coverage.
    3. Partition the Hamiltonian path into contiguous segments using :func:`_partition_path_into_segments` so that each
       segment meets the minimum node length.
    4. Validate that the partitioned segments are simple paths and meet rules using :func:`_validate_segments` and
       :func:`_compute_segment_map`.
    5. Convert segment endpoints into a puzzle grid with :func:`_segments_to_level` and return the resulting
       :class:`numberlink.levels.Level`.

    :param cfg: Generator configuration object. See :class:`numberlink.config.GeneratorConfig` for details.
    :type cfg: GeneratorConfig

    :return: Generated puzzle level as a :class:`numberlink.levels.Level`.
    :rtype: Level

    :raises ValueError: If the grid dimensions or color count are not positive, or if bridge placement is requested in
        the configuration. If the generator fails to produce a valid partition within a bounded number of attempts it
        falls back to :func:`_gen_random_walk` and does not raise here.
    """
    h: int = cfg.height
    w: int = cfg.width
    n_colors: int = cfg.colors

    if h <= 0 or w <= 0:
        raise ValueError("Generator width and height must be positive")
    if n_colors <= 0:
        raise ValueError("Generator requires at least one color")
    if cfg.bridges_probability:
        raise ValueError("Bridge placement is not supported by the procedural generator")

    total_cells: int = h * w
    min_nodes: int = max(3, cfg.min_path_length)
    rng: Generator = np.random.default_rng(cfg.seed)

    max_attempts: int = max(64, n_colors * 16)
    rh_attempts: int = 512  # attempts for random Hamiltonian path
    loop_moves_attempts: int = max(64, total_cells)

    for _ in range(max_attempts):
        path: list[Coord] | None = _random_hamiltonian_path(h, w, rng, attempts=rh_attempts)
        if path is None:
            path = _build_initial_path(h, w, rng)

        # Apply additional randomization to the Hamiltonian path
        _apply_loop_moves(path, rng, attempts=loop_moves_attempts)

        candidate: list[list[Coord]] | None = _partition_path_into_segments(path, n_colors, min_nodes, rng)
        if candidate is None:
            continue

        try:
            _validate_segments(candidate, _compute_segment_map(candidate, h, w), min_nodes)
        except ValueError:
            continue

        rng.shuffle(candidate)
        return _enhance_variability(
            Level(grid=_segments_to_level(candidate, h, w), bridges=None, solution=candidate), rng
        )

    if allow_random_fallback:
        return _gen_random_walk(cfg, variant=variant)

    raise RuntimeError("Hamiltonian generator failed to partition the path into valid segments.")


def _are_adjacent(a: Coord, b: Coord) -> bool:
    """Return whether two coordinates are orthogonally adjacent.

    Adjacency is measured by the Manhattan distance and requires a distance of exactly ``1``.

    :param a: First coordinate as a ``(row, column)`` pair.
    :type a: Coord
    :param b: Second coordinate as a ``(row, column)`` pair.
    :type b: Coord
    :return: ``True`` when the coordinates are adjacent, otherwise ``False``.
    :rtype: bool
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1  # Manhattan distance of 1


def _paths_cover_grid(paths: list[list[Coord]], height: int, width: int) -> bool:
    """Return whether the supplied solution paths cover every cell in the grid.

    :param paths: Collection of solution paths where each path is a list of coordinates.
    :type paths: list[list[Coord]]
    :param height: Grid row count.
    :type height: int
    :param width: Grid column count.
    :type width: int
    :return: ``True`` when every cell in the ``height`` by ``width`` grid is included in `paths`, otherwise ``False``.
    :rtype: bool
    """
    if not paths:
        return False

    covered: set[Coord] = set()
    target: int = height * width
    add: Callable[[Iterable[Coord]], None] = covered.update

    for path in paths:
        add(path)
        if len(covered) == target:  # early exit when full coverage achieved
            return True

    return len(covered) == target


def _build_neighbor_map(height: int, width: int) -> dict[Coord, list[Coord]]:
    """Precompute orthogonal neighbours for every cell in the grid.

    :param height: Number of grid rows.
    :type height: int
    :param width: Number of grid columns.
    :type width: int
    :return: Mapping from a coordinate to a list of orthogonally adjacent coordinates.
    :rtype: dict[Coord, list[Coord]]
    """
    steps: tuple[Coord, ...] = DIR_STEPS
    range_h: range = range(height)
    range_w: range = range(width)

    # return neighbours
    return {
        (r, c): [(nr, nc) for dr, dc in steps if 0 <= (nr := r + dr) < height and 0 <= (nc := c + dc) < width]
        for r in range_h
        for c in range_w
    }


def _random_hamiltonian_path(height: int, width: int, rng: Generator, *, attempts: int = 128) -> list[Coord] | None:
    """Generate a Hamiltonian path using randomized backtracking.

    The search performs Warnsdorff style tie breaking by preferring candidates with the fewest onward moves to improve
    success probability. When all attempts fail the function returns ``None`` and callers may fall back to deterministic
    patterns such as :func:`_build_initial_path`.

    :param height: Grid row count.
    :type height: int
    :param width: Grid column count.
    :type width: int
    :param rng: Random number generator used for stochastic decisions.
    :type rng: numpy.random.Generator
    :param attempts: Number of random start attempts to try.
    :type attempts: int, optional
    :return: Hamiltonian path covering every cell as a list of coordinates, or ``None`` if no path is found.
    :rtype: list[Coord] or None
    """
    total_cells: int = height * width
    neighbours: dict[Coord, list[Coord]] = _build_neighbor_map(height, width)

    max_steps: int = max(total_cells * 64, 4096)

    steps: list[int] = [0]
    n_attempts: int = attempts if attempts > 0 else 1

    for _ in range(n_attempts):
        start: Coord = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        path: list[Coord] = [start]
        visited: set[Coord] = {start}
        steps[0] = 0
        if _hamiltonian_backtrack(path, visited, neighbours, total_cells, max_steps, steps, rng):
            return path

    return None


def _hamiltonian_backtrack(
    path: list[Coord],
    visited: set[Coord],
    neighbours: dict[Coord, list[Coord]],
    total_cells: int,
    max_steps: int,
    steps: list[int],
    rng: Generator,
) -> bool:
    """Backtracking helper used by :func:`_random_hamiltonian_path`.

    The helper advances the recursive search and respects ``max_steps`` to
    avoid excessively long runs. It implements Warnsdorff style tie breaking
    by ordering candidate moves according to the number of onward moves.

    :param path: Current path under construction.
    :type path: list[Coord]
    :param visited: Set of visited coordinates.
    :type visited: set[Coord]
    :param neighbours: Mapping from a coordinate to its orthogonal neighbours.
    :type neighbours: dict[Coord, list[Coord]]
    :param total_cells: Total number of cells to cover.
    :type total_cells: int
    :param max_steps: Maximum allowed search steps to limit runtime.
    :type max_steps: int
    :param steps: Mutable single element list used as a counter for steps.
    :type steps: list[int]
    :param rng: Random number generator used for tie breaking.
    :type rng: numpy.random.Generator
    :return: ``True`` when a Hamiltonian path covering all cells is found.
    :rtype: bool
    """
    steps[0] += 1
    if steps[0] > max_steps:
        return False

    if len(path) == total_cells:
        return True

    current: Coord = path[-1]
    vis: set[Coord] = visited
    neigh: dict[Coord, list[Coord]] = neighbours

    # Degree buckets (onward move counts are in [0, 4] on an orthogonal grid)
    b0: list[Coord] = []
    b1: list[Coord] = []
    b2: list[Coord] = []
    b3: list[Coord] = []
    b4: list[Coord] = []

    cand: Coord
    for cand in neigh[current]:
        if cand in vis:
            continue

        onward = 0
        for nn in neigh[cand]:
            if nn not in vis:
                onward += 1

        if onward == 0:
            b0.append(cand)
        elif onward == 1:
            b1.append(cand)
        elif onward == 2:
            b2.append(cand)
        elif onward == 3:
            b3.append(cand)
        else:
            b4.append(cand)

    if not (b0 or b1 or b2 or b3 or b4):
        return False

    # Process candidates from smallest onward degree to largest
    # Within a bucket, visit in a random order using a single permutation draw
    for bucket in (b0, b1, b2, b3, b4):
        if not bucket:
            continue
        order: NDArray[np.int64] = rng.permutation(len(bucket))
        append: Callable[[Coord], None] = path.append
        pop: Callable[..., Coord] = path.pop
        add: Callable[[Coord], None] = vis.add
        remove: Callable[[Coord], None] = vis.remove

        for idx in order:
            cand = bucket[int(idx)]
            append(cand)
            add(cand)
            if _hamiltonian_backtrack(path, vis, neigh, total_cells, max_steps, steps, rng):
                return True
            remove(cand)
            pop()

    return False


def _shuffle_letter_assignments(level: Level, rng: Generator) -> Level:
    """Randomize which letter is assigned to each solution path.

    The function returns a new :class:`numberlink.levels.Level` with grid characters remapped and with the solution
    paths reordered to match the new assignment. When ``level.solution`` is ``None`` or contains fewer than two paths
    the input ``level`` is returned unchanged.

    :param level: Level to remap.
    :type level: Level
    :param rng: Random number generator used to shuffle assignments.
    :type rng: numpy.random.Generator
    :return: New :class:`numberlink.levels.Level` with shuffled letter assignments.
    :rtype: Level
    """
    solution: list[list[Coord]] | None = level.solution
    if solution is None or len(solution) <= 1:
        return level

    num_colors: int = len(solution)
    order: list[int] = list(range(num_colors))
    rng.shuffle(order)

    # Build inverse mapping old_idx -> new_idx
    inverse: list[int] = [0] * num_colors
    for new_idx, old_idx in enumerate(order):
        inverse[old_idx] = new_idx

    # Precompute letter remaps: letters[old_idx] -> 'A' + inverse[old_idx]
    base_a: int = ord("A")
    letters: list[str] = [chr(base_a + inverse[i]) for i in range(num_colors)]

    # Remap grid rows (dot is special-cased)
    new_grid_rows: list[str] = [
        "".join(("." if ch == "." else letters[ord(ch) - base_a]) for ch in row) for row in level.grid
    ]

    # Reorder and copy solution paths
    new_solution: list[list[Coord]] = [list(solution[idx]) for idx in order]

    # Copy bridges (empty set -> None)
    bridge: Iterable[Coord] | None = level.bridges
    bridges_copy: set[Coord] | None = set(bridge) if bridge else None

    return Level(grid=new_grid_rows, bridges=bridges_copy, solution=new_solution)


def _rotated_dims(height: int, width: int, rotation: int) -> Coord:
    """
    Compute grid dimensions after applying a rotation.

    Determine the resulting ``(height, width)`` pair when a rectangular grid with the supplied ``height`` and ``width``
    is rotated by ``rotation`` steps of 90 degrees clockwise. Rotation values are integers in the range ``0..3`` where
    each increment represents a 90 degree clockwise rotation. When the rotation is odd the dimensions are swapped.

    This helper is used by :func:`_apply_random_orientation` to allocate the transformed grid matrix before mapping
    characters and coordinates. It does not validate the numeric range of ``rotation`` and will return a swapped pair
    for odd values and the original pair for even values.

    :param height: Original grid row count.
    :type height: int
    :param width: Original grid column count.
    :type width: int
    :param rotation: Number of 90 degree clockwise rotations to apply.
    :type rotation: int

    :return: Tuple ``(new_height, new_width)`` after rotation.
    :rtype: :class:`tuple` of two :class:`int`

    .. seealso::
       :func:`_transform_coord` for per-coordinate transformation under the
       same rotation convention.
    """
    if rotation & 1:  # odd -> swap (same as `if rotation % 2 == 1`)
        return width, height

    return height, width


def _transform_coord(r: int, c: int, height: int, width: int, rotation: int) -> Coord:
    """
    Transform a coordinate by a rotation of the grid.

    Map an input coordinate ``(r, c)`` from the original grid to the corresponding coordinate in the grid rotated by
    ``rotation`` steps of 90 degrees clockwise. Rotation values follow the same convention used by :func:`_rotated_dims`
    where ``0`` represents no rotation and ``1`` represents a 90 degree clockwise rotation.

    The mapping rules are:

    - ``rotation == 0``: ``(r, c) -> (r, c)`` (identity)
    - ``rotation == 1``: ``(r, c) -> (c, height - 1 - r)``
    - ``rotation == 2``: ``(r, c) -> (height - 1 - r, width - 1 - c)``
    - ``rotation == 3``: ``(r, c) -> (width - 1 - c, r)``

    This function is used by :func:`_apply_random_orientation` when copying grid characters and when transforming
    solution paths and bridge coordinates.

    :param r: Row index in the original grid.
    :type r: int
    :param c: Column index in the original grid.
    :type c: int
    :param height: Original grid row count.
    :type height: int
    :param width: Original grid column count.
    :type width: int
    :param rotation: Number of 90 degree clockwise rotations to apply.
    :type rotation: int

    :return: Transformed coordinate ``(new_row, new_col)`` in the rotated grid.
    :rtype: :class:`tuple` of two :class:`int`

    :raises ValueError: When ``rotation`` is not one of ``0``, ``1``, ``2``, or
        ``3``.

    .. seealso::
       :func:`_rotated_dims` for computing the dimensions of the rotated
       grid used when allocating the destination matrix.
    """
    if rotation == 0:
        return r, c
    if rotation == 1:
        return c, height - 1 - r
    if rotation == 2:
        return height - 1 - r, width - 1 - c
    if rotation == 3:
        return width - 1 - c, r
    raise ValueError(f"Unsupported rotation {rotation}")


def _apply_random_orientation(level: Level, rng: Generator) -> Level:
    """Randomly rotate and optionally mirror a generated level.

    Apply a random rotation and an independent horizontal flip when selected. The returned
    :class:`numberlink.levels.Level` uses transformed coordinates for the :attr:`numberlink.levels.Level.solution`
    attribute and for the :attr:`numberlink.levels.Level.bridges` attribute when present.

    :param level: Level to transform.
    :type level: :class:`numberlink.levels.Level`
    :param rng: Random number generator used to select rotation and flip.
    :type rng: numpy.random.Generator
    :return: Transformed level with updated ``grid``, ``bridges``, and ``solution``.
    :rtype: :class:`numberlink.levels.Level`

    :raises ValueError: If an unsupported rotation value is passed to
        :func:`_transform_coord`.
    """
    grid: list[str] = level.grid
    if not grid:
        return level

    rotation: int = int(rng.integers(0, 4))
    flip: bool = bool(rng.integers(0, 2))
    if rotation == 0 and not flip:
        return level

    height: int = len(grid)
    width: int = len(grid[0])
    new_height, new_width = _rotated_dims(height, width, rotation)

    # Prebuild a reusable "." row and copy it to form the matrix
    dot_row: list[str] = ["."] * new_width
    new_grid_matrix: list[list[str]] = [dot_row.copy() for _ in range(new_height)]

    # Grid transform
    if flip:
        mirror: int = new_width - 1
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch == ".":
                    continue

                nr, nc = _transform_coord(r, c, height, width, rotation)
                new_grid_matrix[nr][mirror - nc] = ch

    else:
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                if ch == ".":
                    continue
                nr, nc = _transform_coord(r, c, height, width, rotation)
                new_grid_matrix[nr][nc] = ch

    # Solution transform
    new_solution: list[list[Coord]] | None = None
    if level.solution is not None:
        sol: list[list[Coord]] = level.solution
        n_paths: int = len(sol)
        # Create a concrete list to fill, avoid typing as Optional while assigning
        filled_solution: list[list[Coord]] = []
        if flip:
            mirror = new_width - 1
            for i in range(n_paths):
                path: list[Coord] = sol[i]
                transformed_path: list[Coord] = []
                for r, c in path:
                    nr, nc = _transform_coord(r, c, height, width, rotation)
                    transformed_path.append((nr, mirror - nc))
                filled_solution.append(transformed_path)
        else:
            for i in range(n_paths):
                path = sol[i]
                transformed_path = []
                for r, c in path:
                    nr, nc = _transform_coord(r, c, height, width, rotation)
                    transformed_path.append((nr, nc))
                filled_solution.append(transformed_path)

        new_solution = filled_solution

    # Bridges transform
    new_bridges: set[Coord] | None = None
    nb: set[Coord]
    if level.bridges:
        br: Iterable[Coord] = level.bridges
        if flip:
            mirror = new_width - 1
            nb = set()
            for r, c in br:
                nr, nc = _transform_coord(r, c, height, width, rotation)
                nb.add((nr, mirror - nc))
            new_bridges = nb
        else:
            nb = set()
            for r, c in br:
                nr, nc = _transform_coord(r, c, height, width, rotation)
                nb.add((nr, nc))
            new_bridges = nb

    new_grid: list[str] = ["".join(row) for row in new_grid_matrix]
    return Level(grid=new_grid, bridges=new_bridges if new_bridges else None, solution=new_solution)


def _enhance_variability(level: Level, rng: Generator) -> Level:
    """Apply post processing steps that introduce additional visual variety.

    Compose :func:`_apply_random_orientation` and :func:`_shuffle_letter_assignments` to produce a visually varied
    puzzle instance. The returned object preserves the semantics of :class:`numberlink.levels.Level`.

    :param level: Base level to vary.
    :type level: :class:`numberlink.levels.Level`
    :param rng: Random number generator used for stochastic steps.
    :type rng: numpy.random.Generator
    :return: Varied level with normalized types for attributes.
    :rtype: :class:`numberlink.levels.Level`
    """
    varied: Level = _shuffle_letter_assignments(_apply_random_orientation(level, rng), rng)
    return Level(
        grid=varied.grid,
        bridges=None if not varied.bridges else set(varied.bridges),
        solution=None if varied.solution is None else [list(path) for path in varied.solution],
    )


def _build_serpentine_segments(
    height: int, width: int, n_colors: int, min_nodes: int, rng: Generator
) -> list[list[Coord]] | None:
    """Construct a deterministic Hamiltonian path and partition it into segments.

    Use :func:`_build_initial_path` to create a serpentine Hamiltonian path and then partition it into ``n_colors``
    contiguous segments. Each segment meets the ``min_nodes`` requirement when partitioning succeeds.

    :param height: Grid row count.
    :type height: int
    :param width: Grid column count.
    :type width: int
    :param n_colors: Number of segments to produce.
    :type n_colors: int
    :param min_nodes: Minimum allowed nodes in each segment.
    :type min_nodes: int
    :param rng: Random number generator used to preserve call signature.
    :type rng: numpy.random.Generator
    :return: List of segments when partitioning succeeds or ``None`` when it fails.
    :rtype: list[list[Coord]] or None
    """
    path: list[Coord] = _build_initial_path(height, width, rng)

    total_cells: int = len(path)
    base_len: int = total_cells // n_colors
    remainder: int = total_cells % n_colors

    lengths: list[int] = [base_len + (1 if i < remainder else 0) for i in range(n_colors)]
    if any(length < min_nodes for length in lengths):
        return None

    segments: list[list[Coord]] = []
    idx: int = 0
    for length in lengths:
        segments.append(path[idx : idx + length])
        idx += length

    return segments


def _build_initial_path(height: int, width: int, rng: Generator) -> list[Coord]:
    """Create a Hamiltonian path that covers the entire grid using a serpentine sweep.

    Construct a full coverage path by sweeping rows or columns in alternating directions. The initial sweep axis and
    an optional reversal are selected using the provided random number generator.

    :param height: Number of rows in the grid.
    :type height: int
    :param width: Number of columns in the grid.
    :type width: int
    :param rng: Numpy random number generator used to choose orientation and reversal. See :mod:`numpy.random`.
    :type rng: numpy.random.Generator
    :return: List of coordinates that covers every cell exactly once.
    :rtype: list[Coord]
    :raises ValueError: If ``height`` or ``width`` is not positive.
    """
    if height <= 0 or width <= 0:
        raise ValueError("Grid must contain at least one cell")

    # Keep RNG consumption identical to the original
    choice: int = int(rng.integers(4))
    by_rows: bool = (choice & 2) == 0
    reverse: bool = (choice & 1) == 1

    h, w = height, width
    fwd_h, rev_h = range(h), range(h - 1, -1, -1)
    fwd_w, rev_w = range(w), range(w - 1, -1, -1)

    if by_rows:
        # Row sweep: precompute which column order to use for even/odd rows
        rows: range = rev_h if reverse else fwd_h
        cols_by_parity: tuple[range, range] = (rev_w if not reverse else fwd_w, fwd_w if not reverse else rev_w)
        return [(r, c) for r in rows for c in cols_by_parity[r & 1]]

    # Column sweep: precompute which row order to use for even/odd columns
    cols: range = rev_w if reverse else fwd_w
    rows_by_parity: tuple[range, range] = (rev_h if not reverse else fwd_h, fwd_h if not reverse else rev_h)
    return [(r, c) for c in cols for r in rows_by_parity[c & 1]]


def _apply_loop_moves(path: list[Coord], rng: Generator, attempts: int = 128) -> None:
    """Apply random 2-opt style loop reversals to introduce variety while preserving coverage.

    Modify ``path`` in place by performing loop reversals that preserve path contiguity. A reversal is applied
    only when the candidate move keeps the path contiguous according to :func:`_are_adjacent`. Increase ``attempts``
    for more mixing or reduce it for speed.

    :param path: Hamiltonian path represented as a list of ``(row, column)`` coordinates.
    :type path: list[Coord]
    :param rng: Numpy random number generator used to select indices. See :mod:`numpy.random`.
    :type rng: numpy.random.Generator
    :param attempts: Number of random move trials to perform. Defaults to ``128``.
    :type attempts: int, optional
    :return: ``None``. The input ``path`` is modified in place.
    :rtype: None
    """
    n: int = len(path)
    if n < 6:
        return

    for _ in range(attempts):
        # Convert to built-in ints to keep slicing/indexing fast
        i: int = int(rng.integers(1, n - 3))
        j: int = int(rng.integers(i + 1, n - 2))

        a_prev: Coord = path[i - 1]
        a_curr: Coord = path[i]
        b_curr: Coord = path[j]
        b_next: Coord = path[j + 1]

        if not (_are_adjacent(a_prev, b_curr) and _are_adjacent(a_curr, b_next)):
            continue

        ii, jj = i, j
        while ii < jj:
            path[ii], path[jj] = path[jj], path[ii]
            ii += 1
            jj -= 1


def _compute_conflict_min_starts(path: list[Coord]) -> list[int]:
    """Identify the earliest valid segment start index for each position in the Hamiltonian path.

    Compute constraints that prevent partitioned segments from producing adjacent endpoints that belong to different
    segments. For each path index ``i`` the returned list contains the minimum allowed start index for a segment that
    ends at ``i``.

    :param path: Hamiltonian path as a list of coordinates.
    :type path: list[Coord]
    :return: For each index ``i`` the earliest allowed start index for a segment ending at ``i``.
    :rtype: list[int]
    """
    n: int = len(path)
    if n == 0:
        return []

    index_map: dict[Coord, int] = {coord: i for i, coord in enumerate(path)}
    min_start: list[int] = [0] * n
    steps: tuple[Coord, ...] = DIR_STEPS

    for idx in range(n):
        r, c = path[idx]
        limit: int = idx + 1
        # Check 4-neighborhood
        for dr, dc in steps:
            other_idx: int | None = index_map.get((r + dr, c + dc))
            if other_idx is not None and other_idx > limit and min_start[other_idx] < limit:
                min_start[other_idx] = limit

    return min_start


def _partition_choices(
    pos: int,
    segments_left: int,
    total_cells: int,
    min_len: int,
    min_start: list[int],
    rng: Generator,
    memo: dict[Coord, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Compute viable segment endpoints for dynamic partitioning of a Hamiltonian path.

    This recursive helper for :func:`_partition_path_into_segments` uses memoization to avoid repeated work. It
    enforces the minimum segment length and the adjacency constraints produced by :func:`_compute_conflict_min_starts`.

    :param pos: Current path index where a new segment would begin.
    :type pos: int
    :param segments_left: Number of segments remaining to place.
    :type segments_left: int
    :param total_cells: Total number of cells in the Hamiltonian path.
    :type total_cells: int
    :param min_len: Minimum allowed length for a segment.
    :type min_len: int
    :param min_start: Constraint list returned by :func:`_compute_conflict_min_starts`.
    :type min_start: list[int]
    :param rng: Random number generator used to break ties when the caller needs a random choice.
    :type rng: numpy.random.Generator
    :param memo: Memoization mapping keyed by tuples ``(pos, segments_left)``.
    :type memo: dict[Coord, tuple[int, ...] | None]
    :return: Tuple of viable end indices for a segment starting at ``pos`` or ``None`` when no viable choices exist.
    :rtype: tuple[int, ...] or None
    """
    key: Coord = (pos, segments_left)
    res: tuple[int, ...] | None = memo.get(key)
    if res is not None or key in memo:  # handles stored None too
        return res

    result: tuple[int, ...] | None

    # Base cases
    if segments_left == 0:
        result = () if pos == total_cells else None
        memo[key] = result
        return result

    # Not enough cells left to fill required segments
    if total_cells - pos < segments_left * min_len:
        memo[key] = None
        return None

    # We must end by max_end (inclusive) to leave room for remaining segments
    max_end: int = total_cells - (segments_left - 1) * min_len - 1
    viable: list[int] = []

    # We can skip checking ends shorter than min_len, but we must still honor the
    # prefix constraint: max_{e in [pos, end]} min_start[e] <= pos
    max_required_start: int = 0
    start_end: int = pos + min_len - 1

    # Incorporate the skipped prefix [pos, start_end)
    if start_end > pos:
        # manual small loop is fastest for short ranges
        for e in range(pos, start_end):
            max_required_start = max(max_required_start, min_start[e])
        if max_required_start > pos:
            memo[key] = None
            return None

    # Scan candidate ends [start_end .. max_end]
    for end in range(start_end, max_end + 1):
        v: int = min_start[end]
        if v > max_required_start:
            max_required_start = v
            if max_required_start > pos:
                break  # further ends won't help since prefix max only increases

        # Recurse only when the remainder can be partitioned
        if _partition_choices(end + 1, segments_left - 1, total_cells, min_len, min_start, rng, memo) is not None:
            viable.append(end)

    result = tuple(viable) if viable else None
    memo[key] = result
    return result


def _build_partition(
    pos: int,
    segments_left: int,
    total_cells: int,
    min_len: int,
    min_start: list[int],
    rng: Generator,
    memo: dict[Coord, tuple[int, ...] | None],
    segments_bounds: list[Coord],
) -> bool:
    """Recursively construct segment bounds for a Hamiltonian path partition.

    Attempt to place ``segments_left`` contiguous segments starting at ``pos``. For the current position query
    :func:`_partition_choices` to obtain viable end indices and choose one uniformly using ``rng``. Append chosen
    bounds to ``segments_bounds``. The recursion succeeds when the chosen bounds cover ``total_cells``.

    :param pos: Current start index for the next segment.
    :type pos: int
    :param segments_left: Number of segments remaining to place.
    :type segments_left: int
    :param total_cells: Total number of cells in the path.
    :type total_cells: int
    :param min_len: Minimum allowed length of each segment.
    :type min_len: int
    :param min_start: Adjacency constraint list returned by :func:`_compute_conflict_min_starts`.
    :type min_start: list[int]
    :param rng: Random number generator used for uniform tie breaking.
    :type rng: numpy.random.Generator
    :param memo: Memoization mapping shared with :func:`_partition_choices`.
    :type memo: dict[Coord, tuple[int, ...] | None]
    :param segments_bounds: Output list that will receive chosen ``(start, end)`` index tuples.
    :type segments_bounds: list[Coord]
    :return: ``True`` if a complete partition covering ``total_cells`` was constructed, otherwise ``False``.
    :rtype: bool
    """
    while segments_left:
        options: tuple[int, ...] | None = _partition_choices(
            pos, segments_left, total_cells, min_len, min_start, rng, memo
        )
        if not options:
            return False

        # Choose uniformly among viable ends
        end_idx: int = options[int(rng.integers(len(options)))]
        segments_bounds.append((pos, end_idx))
        pos = end_idx + 1
        segments_left -= 1

    return pos == total_cells


def _partition_path_into_segments(
    path: list[Coord], n_segments: int, min_len: int, rng: Generator
) -> list[list[Coord]] | None:
    """Split a Hamiltonian path into contiguous segments while avoiding adjacent endpoints.

    Attempt to partition ``path`` into ``n_segments`` contiguous subpaths. Each subpath has at least ``min_len``
    nodes. The endpoints of different segments are not adjacent on the grid. The partitioning uses
    :func:`_partition_choices` and the adjacency constraints from :func:`_compute_conflict_min_starts`.

    :param path: Hamiltonian path that covers every grid cell.
    :type path: list[Coord]
    :param n_segments: Desired number of segments.
    :type n_segments: int
    :param min_len: Minimum allowed length of each segment.
    :type min_len: int
    :param rng: Random number generator used to break ties when multiple partitions are possible.
    :type rng: numpy.random.Generator
    :return: List of segments when partitioning succeeds or ``None`` when no valid partition exists.
    :rtype: list[list[Coord]] or None
    """
    total_cells: int = len(path)
    if n_segments <= 0 or total_cells < n_segments * min_len:
        return None

    min_start: list[int] = _compute_conflict_min_starts(path)
    memo: dict[Coord, tuple[int, ...] | None] = {}
    segments_bounds: list[Coord] = []

    if not _build_partition(0, n_segments, total_cells, min_len, min_start, rng, memo, segments_bounds):
        return None

    # Slice once per segment
    return [path[s : e + 1] for s, e in segments_bounds]


def _compute_segment_map(segments: list[list[Coord]], height: int, width: int) -> NDArray[np.int16]:
    """Compute a 2D numpy array that maps each grid cell to its segment index.

    The returned array uses ``-1`` to indicate unassigned cells. Raise :exc:`ValueError` when a segment contains a
    coordinate that is outside the grid bounds or when two segments assign the same cell.

    :param segments: List of segments where each segment is a list of coordinates.
    :type segments: list[list[Coord]]
    :param height: Grid row count.
    :type height: int
    :param width: Grid column count.
    :type width: int
    :return: Numpy array with shape ``(height, width)`` mapping each cell to its segment index.
    :rtype: numpy.ndarray
    :raises ValueError: If a segment cell is outside grid bounds or if segments overlap on the same cell.
    """
    seg_map: NDArray[np.int16] = np.full((height, width), -1, dtype=np.int16)
    for idx, segment in enumerate(segments):
        for r, c in segment:
            if not (0 <= r < height and 0 <= c < width):
                raise ValueError("Segment cell outside of grid bounds")

            if seg_map[r, c] != -1:
                raise ValueError("Segments overlap on the same cell")

            seg_map[r, c] = idx

    return seg_map


def _validate_segments(segments: list[list[Coord]], seg_map: NDArray[np.int16], min_nodes: int) -> None:
    """Validate that every segment is a simple path and meets NumberLink constraints.

    Perform the following checks. Each segment length is at least ``max(3, min_nodes)``. Each segment is contiguous as
    defined by :func:`_are_adjacent`. Segment endpoints are not adjacent to each other. Using ``seg_map`` ensure that
    each grid cell is assigned and that each segment has exactly two endpoints when measured on the grid.

    :param segments: Partitioned segments to validate.
    :type segments: list[list[Coord]]
    :param seg_map: Grid mapping produced by :func:`_compute_segment_map`.
    :type seg_map: numpy.ndarray
    :param min_nodes: Minimum nodes a single segment must contain. Effective minimum is ``max(3, min_nodes)``.
    :type min_nodes: int
    :return: ``None`` when validation succeeds.
    :rtype: None
    :raises ValueError: When any validation rule fails.
    """
    height: int
    width: int
    height, width = seg_map.shape
    idx: int
    for idx, segment in enumerate(segments):
        if len(segment) < max(3, min_nodes):
            raise ValueError(f"Segment {idx} is shorter than the minimum requirement")

        # Check contiguity and adjacency distances
        for a, b in zip(segment, segment[1:], strict=False):
            if not _are_adjacent(a, b):
                raise ValueError(f"Segment {idx} is not contiguous")

        start: Coord = segment[0]
        end: Coord = segment[-1]
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) <= 1:
            raise ValueError(f"Segment {idx} endpoints are adjacent or identical")

    # Grid-level endpoint degree checks
    endpoint_counts: dict[int, int] = dict.fromkeys(range(len(segments)), 0)
    for r in range(height):
        for c in range(width):
            idx = int(seg_map[r, c])
            if idx == -1:
                raise ValueError("Unassigned cell detected during validation")

            same_seg_neigh = 0
            # Check 4 neighbors
            nr: int = r - 1
            if nr >= 0 and seg_map[nr, c] == idx:
                same_seg_neigh += 1

            nr = r + 1
            if nr < height and seg_map[nr, c] == idx:
                same_seg_neigh += 1

            nc: int = c - 1
            if nc >= 0 and seg_map[r, nc] == idx:
                same_seg_neigh += 1

            nc = c + 1
            if nc < width and seg_map[r, nc] == idx:
                same_seg_neigh += 1

            if same_seg_neigh == 1:
                endpoint_counts[idx] += 1

            elif same_seg_neigh == 2:
                # ok, interior point of a simple path
                pass

            else:
                # 0 or >2 neighbors in the same segment -> dead end or branch
                raise ValueError("Path branches or dead-ends detected. Invalid configuration")

    for idx, count in endpoint_counts.items():
        if count != 2:
            raise ValueError(f"Segment {idx} does not have exactly two endpoints")


def _segments_to_level(segments: list[list[Coord]], height: int, width: int) -> list[str]:
    """Render segment endpoints onto the puzzle grid.

    Assign an uppercase letter to each segment starting at ``'A'``. Place only the endpoints of each segment on the
    returned grid. Return a list of strings where each string represents a row and the dot character represents an
    empty cell.

    :param segments: List of segments where each segment is a list of coordinates.
    :type segments: list[list[Coord]]
    :param height: Grid row count.
    :type height: int
    :param width: Grid column count.
    :type width: int
    :return: Puzzle grid as a list of row strings.
    :rtype: list[str]
    """
    grid_chars: list[list[str]] = [["." for _ in range(width)] for _ in range(height)]
    for idx, segment in enumerate(segments):
        letter: str = chr(ord("A") + idx)
        start: Coord = segment[0]
        end: Coord = segment[-1]
        grid_chars[start[0]][start[1]] = letter
        grid_chars[end[0]][end[1]] = letter

    return ["".join(row) for row in grid_chars]


def _shortest_dist(a: Coord, b: Coord, allow_diagonal: bool = False) -> int:
    """Compute the shortest grid distance between two coordinates using the configured movement rules.

    Return the Chebyshev distance when ``allow_diagonal`` is ``True``. Otherwise return the Manhattan distance.

    :param a: First coordinate as a ``(row, column)`` pair.
    :type a: Coord
    :param b: Second coordinate as a ``(row, column)`` pair.
    :type b: Coord
    :param allow_diagonal: Whether diagonal movement is permitted.
    :type allow_diagonal: bool
    :return: Shortest distance between ``a`` and ``b`` according to the configured movement rules.
    :rtype: int
    """
    dr: int = abs(a[0] - b[0])
    dc: int = abs(a[1] - b[1])
    if not allow_diagonal:
        return dr + dc  # Manhattan

    return max(dr, dc)  # Chebyshev


def _sample_start_cell(used_endpoints: set[Coord], h: int, w: int, rng: Generator) -> Coord:
    """Sample a start cell uniformly among cells not in ``used_endpoints``.

    :param used_endpoints: Set of coordinates already used as endpoints.
    :type used_endpoints: set[Coord]
    :param h: Grid row count.
    :type h: int
    :param w: Grid column count.
    :type w: int
    :param rng: Random number generator used for uniform sampling.
    :type rng: numpy.random.Generator
    :return: Sampled coordinate that is not in ``used_endpoints``.
    :rtype: Coord
    """
    while True:
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        cell: Coord = (r, c)
        if cell not in used_endpoints:
            return cell


def _gen_random_walk(cfg: GeneratorConfig, *, variant: VariantConfig) -> Level:
    """Generate a puzzle by placing endpoints using per-color random walks.

    For each color the algorithm performs the following steps

    1. Choose a random start cell that is not already used as an endpoint.
    2. Perform a random walk that avoids cells occupied by previous walks.
    3. End at a different cell when the walk satisfies the minimum length requirement.
    4. Ensure the shortest path distance between endpoints meets the configured minimum.

    Only endpoints are placed on the returned grid. Intermediate walk cells are marked as occupied to prevent other
    colors from using them. When ``cfg.bridges_probability`` is greater than ``0.0`` the function may return a set of
    bridge coordinates in the :class:`numberlink.levels.Level` result.

    :param cfg: Generator configuration object. See :class:`numberlink.config.GeneratorConfig`.
    :type cfg: :class:`numberlink.config.GeneratorConfig`
    :return: Generated puzzle level with endpoint letters and optional bridges.
    :rtype: :class:`numberlink.levels.Level`
    :raises ValueError: If grid dimensions or color count are invalid, if the configured minimum path length is too
        large for the board, or if the generator cannot place valid endpoints after the configured number of attempts.
    """
    h, w, n_colors = cfg.height, cfg.width, cfg.colors
    if h <= 0 or w <= 0:
        raise ValueError("Generator width and height must be positive")
    if n_colors < 1:
        raise ValueError("Generator requires at least one color")
    if n_colors > h * w // 2:
        raise ValueError("Too many colors for the board area")

    min_dist: int = max(2, cfg.min_path_length)
    total_cells: int = h * w
    if min_dist > total_cells // 2:
        raise ValueError(f"min_path_length {min_dist} too large for {h}x{w} board")

    rng: Generator = np.random.default_rng(cfg.seed)

    letters: list[str] = [chr(ord("A") + i) for i in range(n_colors)]
    must_fill: bool = variant.must_fill
    max_global_attempts: int = max(1, min(cfg.max_retries, 20)) if must_fill else max(1, cfg.max_retries)
    attempts_per_color: int = 200 if must_fill else 1000

    # Movement directions + precomputed neighbors
    base_dirs: list[Coord] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    if variant.allow_diagonal:
        base_dirs.extend([(-1, -1), (-1, 1), (1, 1), (1, -1)])
    directions: tuple[Coord, ...] = tuple(base_dirs)

    # Precompute neighbors for every cell (row-major indexing)
    neighbors: list[list[tuple[Coord, ...]]] = [[() for _ in range(w)] for _ in range(h)]
    for r in range(h):
        row_nbrs: list[tuple[Coord, ...]] = []
        for c in range(w):
            nbrs: list[Coord] = []
            for dr, dc in directions:
                nr: int = r + dr
                nc: int = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nbrs.append((nr, nc))

            row_nbrs.append(tuple(nbrs))

        neighbors[r] = row_nbrs

    for _ in range(max_global_attempts):
        grid: NDArray[np.str_] = np.full((h, w), ".", dtype="<U1")
        occupied_by_walks: set[Coord] = set()
        used_endpoints: set[Coord] = set()
        solution_paths: list[list[Coord]] = []
        success_all: bool = True

        for color_idx in range(n_colors):
            letter: str = letters[color_idx]
            max_attempts: int = attempts_per_color
            success: bool = False

            for _ in range(max_attempts):
                if total_cells - len(used_endpoints) < 2:
                    success_all = False
                    break

                start_cell: Coord = _sample_start_cell(used_endpoints, h, w, rng)

                min_walk_len: int = min_dist
                free_remaining: int = total_cells - len(occupied_by_walks) - len(used_endpoints) - 2
                max_walk_len: int = min(max(min_dist + 5, total_cells // (n_colors + 1)), free_remaining)

                if max_walk_len < min_walk_len:
                    continue

                walk_length: int = max_walk_len if must_fill else int(rng.integers(min_walk_len, max_walk_len + 1))

                current: Coord = start_cell
                visited_unique: set[Coord] = {current}
                path_stack: list[Coord] = [current]
                index_map: dict[Coord, int] = {current: 0}
                valid_walk: bool = True

                for _ in range(walk_length):
                    # Gather valid neighbor cells
                    cell_nbrs: tuple[Coord, ...] = neighbors[current[0]][current[1]]
                    valid_nbrs: list[Coord] = []

                    # (A and B) or (C and D)
                    if len(path_stack) > 1:
                        for nbr in cell_nbrs:
                            if (nbr not in occupied_by_walks and nbr not in used_endpoints) or (nbr == start_cell):
                                valid_nbrs.append(nbr)
                    else:
                        for nbr in cell_nbrs:
                            if (nbr not in occupied_by_walks) and (nbr not in used_endpoints):
                                valid_nbrs.append(nbr)

                    if not valid_nbrs:
                        if len(path_stack) < min_dist:
                            valid_walk = False
                        break

                    # Choose uniformly among valid neighbors
                    current = valid_nbrs[int(rng.integers(0, len(valid_nbrs)))]

                    # Cycle handling (trim looped segment)
                    prev_idx: int | None = index_map.get(current)
                    if prev_idx is not None:
                        # Remove indices for the truncated tail
                        for removed in path_stack[prev_idx + 1 :]:
                            index_map.pop(removed, None)

                        path_stack = path_stack[: prev_idx + 1]

                    else:
                        index_map[current] = len(path_stack)
                        path_stack.append(current)

                    visited_unique.add(current)

                end_cell: Coord = current

                # Early lower-bound check: if LB >= min_dist, the shortest path is guaranteed >= min_dist
                need_exact_dist: bool = _shortest_dist(start_cell, end_cell, variant.allow_diagonal) < min_dist
                exact_ok: bool = (
                    _shortest_dist(start_cell, end_cell, variant.allow_diagonal) >= min_dist
                    if need_exact_dist
                    else True
                )

                if (
                    not valid_walk
                    or len(path_stack) < min_dist
                    or (must_fill and len(path_stack) - 1 < walk_length)
                    or end_cell == start_cell
                    or end_cell in used_endpoints
                    or not exact_ok
                ):
                    continue

                grid[start_cell] = letter
                grid[end_cell] = letter
                used_endpoints.add(start_cell)
                used_endpoints.add(end_cell)

                for cell in visited_unique:
                    if cell not in {start_cell, end_cell}:
                        occupied_by_walks.add(cell)

                solution_paths.append(list(path_stack))
                success = True
                break

            if not success:
                success_all = False
                break

        if not success_all:
            continue

        if must_fill and not _paths_cover_grid(solution_paths, h, w):
            continue

        bridges: set[Coord] | None = None
        if cfg.bridges_probability > 0.0:
            endpoint_set: set[Coord] = used_endpoints
            candidate_cells: set[Coord] = {cell for path in solution_paths for cell in path if cell not in endpoint_set}
            selected: set[Coord] = {cell for cell in candidate_cells if rng.random() < cfg.bridges_probability}
            bridges = selected or None

        base_level: Level = Level(grid=["".join(row) for row in grid], bridges=bridges, solution=solution_paths)
        return _enhance_variability(base_level, rng)

    if must_fill:
        # Fall back to Hamiltonian partition generation to guarantee a must-fill puzzle.
        fallback_cfg: GeneratorConfig = replace(cfg, mode="hamiltonian", bridges_probability=0.0)
        fallback_level: Level | None
        try:
            fallback_level = _gen_hamiltonian_partition(fallback_cfg, variant=variant, allow_random_fallback=False)
        except RuntimeError:
            fallback_level = None
        else:
            segments_for_level: list[list[Coord]] = fallback_level.solution or []
            if cfg.bridges_probability > 0.0:
                endpoint_set = {coord for path in segments_for_level for coord in (path[0], path[-1])}
                candidate_cells = {cell for path in segments_for_level for cell in path if cell not in endpoint_set}
                selected_bridges: set[Coord] = {
                    cell for cell in candidate_cells if rng.random() < cfg.bridges_probability
                }
                enhanced_level = Level(
                    grid=list(fallback_level.grid), bridges=selected_bridges or None, solution=fallback_level.solution
                )
                return _enhance_variability(enhanced_level, rng)

            return fallback_level

        serpentine_segments: list[list[Coord]] | None = _build_serpentine_segments(h, w, n_colors, min_dist, rng)
        if serpentine_segments is not None:
            rng.shuffle(serpentine_segments)
            puzzle_rows: list[str] = _segments_to_level(serpentine_segments, h, w)
            if cfg.bridges_probability > 0.0:
                endpoint_set = {coord for path in serpentine_segments for coord in (path[0], path[-1])}
                candidate_cells = {cell for path in serpentine_segments for cell in path if cell not in endpoint_set}
                selected_bridges = {cell for cell in candidate_cells if rng.random() < cfg.bridges_probability}
                base_level = Level(grid=puzzle_rows, bridges=selected_bridges or None, solution=serpentine_segments)
                return _enhance_variability(base_level, rng)

            base_level = Level(grid=puzzle_rows, bridges=None, solution=serpentine_segments)
            return _enhance_variability(base_level, rng)

    raise ValueError(
        "Could not generate a valid random-walk level that satisfies configuration constraints."
        if must_fill
        else "Could not generate a valid random-walk level with the provided configuration."
    )
