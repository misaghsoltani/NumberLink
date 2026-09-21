from __future__ import annotations

import pytest

from numberlink import NumberLinkRGBEnv, VariantConfig

LARGE_SIDE: int = 130
"""Board side length above the ``int8`` range once used for neighbor arithmetic."""


def _single_column_board(height: int, width: int) -> list[str]:
    """Return a board whose only color has its endpoints in the first and last row of column zero.

    Args:
        height: Number of rows.
        width: Number of columns.

    Returns:
        Row oriented grid strings.
    """
    rows: list[str] = ["." * width for _ in range(height)]
    rows[0] = "A" + "." * (width - 1)
    rows[height - 1] = "A" + "." * (width - 1)
    return rows


@pytest.mark.slow
def test_cell_switching_solves_board_taller_than_int8() -> None:
    """A vertical path across more than 127 rows is recognized as connected."""
    rows: list[str] = _single_column_board(LARGE_SIDE, 3)
    env = NumberLinkRGBEnv(
        render_mode=None, grid=rows, variant=VariantConfig(must_fill=False, cell_switching_mode=True)
    )
    try:
        env.reset(seed=0)
        assert env.H == LARGE_SIDE
        for row in range(1, LARGE_SIDE - 1):
            env.step(env.encode_cell_switching_action(row, 0, 1))
        assert env.is_solved() is True
    finally:
        env.close()


@pytest.mark.slow
def test_cell_switching_rejects_broken_path_on_large_board() -> None:
    """Leaving a gap in the path across more than 127 rows keeps the board unsolved."""
    rows: list[str] = _single_column_board(LARGE_SIDE, 3)
    env = NumberLinkRGBEnv(
        render_mode=None, grid=rows, variant=VariantConfig(must_fill=False, cell_switching_mode=True)
    )
    try:
        env.reset(seed=0)
        for row in range(1, LARGE_SIDE - 1):
            if row == LARGE_SIDE // 2:
                continue
            env.step(env.encode_cell_switching_action(row, 0, 1))
        assert env.is_solved() is False
    finally:
        env.close()


@pytest.mark.slow
def test_path_mode_steps_across_board_wider_than_int8() -> None:
    """Path mode connects endpoints across more than 127 columns."""
    rows: list[str] = ["A" + "." * (LARGE_SIDE - 2) + "A", "." * LARGE_SIDE, "." * LARGE_SIDE]
    env = NumberLinkRGBEnv(render_mode=None, grid=rows, variant=VariantConfig(must_fill=False))
    try:
        env.reset(seed=0)
        assert env.W == LARGE_SIDE
        right: int = next(index for index, vec in enumerate(env.dirs) if (int(vec[0]), int(vec[1])) == (0, 1))
        terminated: bool = False
        for _ in range(LARGE_SIDE):
            _obs, _reward, terminated, truncated, _info = env.step(right)
            if terminated or truncated:
                break
        assert terminated is True
    finally:
        env.close()
