"""Type aliases and small helpers used across the :mod:`numberlink` package.

Provide commonly used aliases such as :class:`numberlink.types.Coord` and the helper functions used to select the
smallest appropriate :mod:`numpy` integer dtype for a numeric range.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

RenderMode: TypeAlias = Literal["rgb_array", "ansi", "human"]
"""Rendering mode for a gym environment observation.

Accepted values are ``"rgb_array"``, ``"ansi"``, and ``"human"``.
"""

Coord: TypeAlias = tuple[int, int]
"""Grid coordinate as a ``(row, column)`` integer pair.

Used throughout the package to identify cell positions in the board.
"""

Lane: TypeAlias = Literal["n", "v", "h"]
"""Lane indicator used in :class:`numberlink.types.CellLane`.

Value ``'n'`` denotes a normal cell, ``'v'`` denotes a vertical bridge lane,
and ``'h'`` denotes a horizontal bridge lane.
"""

CellLane: TypeAlias = tuple[int, int, Lane]
"""Cell coordinate together with a lane indicator as ``(row, column, lane)``.

The third element is one of the values defined by :class:`numberlink.types.Lane`.
"""

RGBInt: TypeAlias = tuple[int, int, int]
"""RGB color represented as ``(red, green, blue)`` integer components.

Each component is in the range ``0..255`` when used for rendering.
"""

ActType: TypeAlias = np.integer | int
"""Action integer type accepted by the environment step method.

May be a :class:`numpy.integer` instance or a built in ``int``.
"""

ObsType: TypeAlias = NDArray[np.uint8]
"""Observation array type returned by :meth:`numberlink.env.NumberLinkRGBEnv.render`.

The array has dtype ``numpy.uint8`` and represents an RGB image.
"""


class Snapshot(TypedDict):
    """TypedDict describing the runtime snapshot captured by :meth:`numberlink.viewer.NumberLinkViewer._snapshot_state`.

    The snapshot includes internal arrays and viewer selection state that are used by
    :meth:`numberlink.viewer.NumberLinkViewer._restore_state` to restore a captured runtime configuration.
    """

    _grid_codes: NDArray[np.integer]
    _lane_v: NDArray[np.integer]
    _lane_h: NDArray[np.integer]
    _heads: list[list[Coord]]
    # Stacks contain CellLane tuples (row, col, lane) per the runtime representation used by `env.NumberLinkRGBEnv`.
    _stacks: list[list[list[CellLane]]]
    _closed: NDArray[np.bool_]
    _steps: int
    sel_color: int
    sel_head: int
    switch_mode: bool
    cursor: list[int]


def select_signed_dtype(min_value: int, max_value: int) -> type[np.signedinteger]:
    """Return the smallest signed :mod:`numpy` integer dtype that covers a range.

    Inspect the absolute bounds of ``min_value`` and ``max_value`` and return one of ``numpy.int8``, ``numpy.int16``,
    ``numpy.int32``, or ``numpy.int64`` depending on which type can represent the full range.

    :param min_value: Minimum value in the range.
    :type min_value: int
    :param max_value: Maximum value in the range.
    :type max_value: int

    :return: Appropriate :mod:`numpy` signed integer dtype type.
    :rtype: type[numpy.signedinteger]
    """
    max_abs: int = max(abs(int(min_value)), abs(int(max_value)))
    if max_abs <= np.iinfo(np.int8).max:
        return np.int8
    if max_abs <= np.iinfo(np.int16).max:
        return np.int16
    if max_abs <= np.iinfo(np.int32).max:
        return np.int32
    return np.int64


def select_unsigned_dtype(max_value: int) -> type[np.unsignedinteger]:
    """Return the smallest unsigned :mod:`numpy` integer dtype that covers ``0``..``max_value``.

    Select among ``numpy.uint8``, ``numpy.uint16``, ``numpy.uint32``, and ``numpy.uint64`` based on the provided maximum
    value. Values below ``0`` are clamped to ``0`` before selection.

    :param max_value: Maximum value in the range.
    :type max_value: int

    :return: Appropriate :mod:`numpy` unsigned integer dtype type.
    :rtype: type[numpy.unsignedinteger]
    """
    max_value = max(0, int(max_value))
    if max_value <= np.iinfo(np.uint8).max:
        return np.uint8
    if max_value <= np.iinfo(np.uint16).max:
        return np.uint16
    if max_value <= np.iinfo(np.uint32).max:
        return np.uint32
    return np.uint64


__all__: list[str] = [
    "RenderMode",
    "Coord",
    "Lane",
    "CellLane",
    "RGBInt",
    "ActType",
    "ObsType",
    "select_signed_dtype",
    "select_unsigned_dtype",
    "Snapshot",
]
