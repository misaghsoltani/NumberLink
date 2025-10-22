"""Level definitions for the :class:`numberlink.env.NumberLinkRGBEnv` environment.

This module defines the :class:`numberlink.levels.Level` dataclass and the :data:`numberlink.levels.LEVELS` mapping of
built-in level definitions. Each :class:`numberlink.levels.Level` stores a row oriented grid where uppercase letters
denote endpoint cells and ``'.'`` denotes empty cells. Optional bridge coordinates may be supplied to mark bridge cells
used by level construction in :mod:`numberlink`.

Use :class:`numberlink.levels.Level` to define custom levels or access :data:`numberlink.levels.LEVELS` for examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources as importlib_resources
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from importlib.resources.abc import Traversable

    from .types import Coord


@dataclass(slots=True)
class Level:
    """Static representation of a NumberLink puzzle board.

    Represent a board as a list of row strings. Uppercase letters denote endpoint cells and ``'.'`` denotes an empty
    cell. Optionally include a sequence of bridge coordinates to indicate bridge cells. An optional precomputed solution
    can be attached for O(1) retrieval by the viewer or tools.

    :param grid: Row oriented grid where each string represents a board row.
    :type grid: list[str]
    :param bridges: Coordinates treated as bridge cells by :func:`numberlink.level_setup.build_level_template` when
        building an environment instance. May be ``None`` when not used.
    :type bridges: collections.abc.Iterable[tuple[int, int]] or None
    :param solution: Optional list of paths where each path is a list of ``(row, column)`` coordinate tuples. Path order
        corresponds to the color order implied by the grid letters.
    :type solution: list[list[tuple[int, int]]] or None
    """

    grid: list[str]
    bridges: Iterable[Coord] | None = None
    solution: list[list[Coord]] | None = None


def _read_text_lines(res_file: Traversable) -> list[str]:
    """Read non-empty, stripped lines from a package resource file."""
    text: str = res_file.read_text(encoding="utf-8")
    rows: list[str] = []
    for line in text.splitlines():
        row: str = line.strip()
        if row:
            rows.append(row)
    return rows


def _load_solution_json(res_file: Traversable) -> list[list[Coord]] | None:
    """Attempt to load a JSON solution adjacent to a grid file.

    Expected JSON format: list of paths, where each path is a list of ``[row, col]`` pairs.
    Returns ``None`` when the file doesn't exist.
    """
    try:
        text: str = res_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    data: list[list[list[int]]] = json.loads(text)
    sol: list[list[Coord]] = []
    for path_list in data:
        coords: list[Coord] = []
        for entry in path_list:
            coords.append((int(entry[0]), int(entry[1])))
        sol.append(coords)
    return sol


def _level_from_asset(txt_name: str) -> Level:
    """Create a Level from a packaged ``assets/levels/*.txt`` resource.

    Also attempts to load an optional ``.sol.json`` with the same stem for solution coordinates.
    """
    base: Traversable = importlib_resources.files(__package__) / "assets" / "levels"
    txt_res: Traversable = base / txt_name
    grid_rows: list[str] = _read_text_lines(txt_res)
    stem: str = txt_name[:-4] if txt_name.endswith(".txt") else txt_name
    sol_res: Traversable = base / f"{stem}.sol.json"
    solution: list[list[Coord]] | None = _load_solution_json(sol_res)
    return Level(grid=grid_rows, bridges=None, solution=solution)


def load_level_from_file(txt_path: str | Path) -> Level:
    """Load a level from a text file with an optional adjacent solution.

    The grid file must contain one row per line. Leading/trailing whitespace and empty lines are ignored.
    If a solution file is present alongside the grid with suffix ``.sol.json`` appended to the grid filename,
    it is parsed and attached to the returned :class:`Level`.

    Example: ``/path/to/board.txt`` with optional ``/path/to/board.txt.sol.json``.
    """
    p: Path = Path(txt_path)
    rows: list[str] = []
    text: str = p.read_text(encoding="utf-8")
    for line in text.splitlines():
        row: str = line.strip()
        if row:
            rows.append(row)
    sol_path: Path = p.with_suffix(p.suffix + ".sol.json")
    solution: list[list[Coord]] | None = None
    if sol_path.exists():
        raw: str = sol_path.read_text(encoding="utf-8")
        data: list[list[list[int]]] = json.loads(raw)
        solution = [[(int(rc[0]), int(rc[1])) for rc in path] for path in data]
    return Level(grid=rows, bridges=None, solution=solution)


def _scan_asset_levels(limit: int = 20) -> dict[str, Level]:
    """Load up to ``limit`` levels from packaged assets into a mapping.

    Level ids are derived from the filename stem with a ``file_`` prefix, for example ``file_5x5_01``.
    """
    base: Traversable = importlib_resources.files(__package__) / "assets" / "levels"
    levels: dict[str, Level] = {}
    names: list[str] = []
    try:
        for entry in base.iterdir():
            name: str = entry.name
            if name.endswith(".txt"):
                names.append(name)
    except Exception:
        names = []
    names.sort()
    for name in names[:limit]:
        stem: str = name[:-4] if name.endswith(".txt") else name
        level_id: str = f"file_{stem}"
        try:
            levels[level_id] = _level_from_asset(name)
        except Exception:
            # Skip malformed assets
            continue
    return levels


# Built-in levels
BUILTIN_LEVELS: dict[str, Level] = {
    "builtin_5x5_rw_4c": Level(
        grid=["..DA.", "D....", "....A", "B..BC", "..C.."],
        bridges=None,
        solution=[
            [(2, 4), (2, 3), (1, 3), (0, 3)],
            [(3, 3), (3, 2), (3, 1), (3, 0)],
            [(3, 4), (4, 4), (4, 3), (4, 2)],
            [(1, 0), (1, 1), (0, 1), (0, 2)],
        ],
    ),
    "builtin_6x6_rw_5c": Level(
        grid=["C...AE", ".AC...", "..D...", "..B.E.", "D...B.", "......"],
        bridges=None,
        solution=[
            [(1, 1), (1, 2), (1, 3), (0, 3), (0, 4)],
            [(3, 2), (3, 3), (4, 3), (4, 4)],
            [(1, 2), (0, 2), (0, 1), (0, 0)],
            [(4, 0), (4, 1), (3, 1), (3, 2), (2, 2)],
            [(0, 5), (1, 5), (2, 5), (3, 5), (3, 4)],
        ],
    ),
    "builtin_7x7_ham_6c": Level(
        grid=[".ED.F..", "..AD.C.", "..B..A.", "..CB...", ".......", ".......", "...EF.."],
        bridges=None,
        solution=[
            [(1, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (4, 5), (3, 5), (2, 5)],
            [(3, 3), (2, 3), (2, 2)],
            [(1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (4, 3), (4, 2), (3, 2)],
            [(0, 2), (0, 3), (1, 3)],
            [(6, 3), (6, 2), (6, 1), (6, 0), (5, 0), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1)],
            [(0, 4), (0, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (6, 5), (6, 4)],
        ],
    ),
    "builtin_8x8_rw_6c": Level(
        grid=["..BF....", "....B.EF", "...A....", ".......E", "..AC....", ".D..DC..", "........", "........"],
        bridges=None,
        solution=[
            [(2, 3), (2, 2), (3, 2), (4, 2)],
            [(0, 2), (0, 3), (1, 3), (1, 4)],
            [(4, 3), (4, 4), (4, 5), (5, 5)],
            [(5, 1), (5, 2), (5, 3), (5, 4)],
            [(1, 6), (2, 6), (2, 7), (3, 7)],
            [(0, 3), (0, 4), (0, 5), (1, 5), (1, 6), (1, 7)],
        ],
    ),
    "builtin_10x10_ham_8c": Level(
        grid=[
            ".E.....B..",
            ".F.AG..EH.",
            "..........",
            "...CA.....",
            "....CGD...",
            ".....DF...",
            "..........",
            "..........",
            "H.........",
            "B.........",
        ],
        bridges=None,
        solution=[
            [(1, 3), (2, 3), (2, 4), (3, 4)],
            [
                (0, 7),
                (0, 8),
                (0, 9),
                (1, 9),
                (2, 9),
                (3, 9),
                (4, 9),
                (5, 9),
                (6, 9),
                (7, 9),
                (8, 9),
                (9, 9),
                (9, 8),
                (9, 7),
                (9, 6),
                (9, 5),
                (9, 4),
                (9, 3),
                (9, 2),
                (9, 1),
                (9, 0),
            ],
            [(3, 3), (4, 3), (4, 4)],
            [
                (5, 5),
                (5, 4),
                (5, 3),
                (5, 2),
                (4, 2),
                (3, 2),
                (2, 2),
                (1, 2),
                (0, 2),
                (0, 3),
                (0, 4),
                (0, 5),
                (0, 6),
                (1, 6),
                (2, 6),
                (3, 6),
                (4, 6),
            ],
            [
                (1, 7),
                (2, 7),
                (3, 7),
                (4, 7),
                (5, 7),
                (6, 7),
                (7, 7),
                (7, 6),
                (7, 5),
                (7, 4),
                (7, 3),
                (7, 2),
                (7, 1),
                (7, 0),
                (6, 0),
                (5, 0),
                (4, 0),
                (3, 0),
                (2, 0),
                (1, 0),
                (0, 0),
                (0, 1),
            ],
            [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6)],
            [(4, 5), (3, 5), (2, 5), (1, 5), (1, 4)],
            [
                (8, 0),
                (8, 1),
                (8, 2),
                (8, 3),
                (8, 4),
                (8, 5),
                (8, 6),
                (8, 7),
                (8, 8),
                (7, 8),
                (6, 8),
                (5, 8),
                (4, 8),
                (3, 8),
                (2, 8),
                (1, 8),
            ],
        ],
    ),
}


# Public mapping of available levels
LEVELS: dict[str, Level] = {**BUILTIN_LEVELS, **_scan_asset_levels(limit=20)}
