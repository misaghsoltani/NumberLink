from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import pytest

from numberlink import LEVELS, GeneratorConfig, NumberLinkRGBEnv, VariantConfig
from numberlink.cli import main as cli_main

if TYPE_CHECKING:
    import pathlib


def test_render_text_includes_header_and_grid() -> None:
    """The ansi rendering starts with a status header, a rule as wide as the header, and one line per row."""
    env = NumberLinkRGBEnv(render_mode="ansi", level_id="builtin_5x5_rw_4c")
    try:
        env.reset(seed=0)
        text: str = env.render_text()
    finally:
        env.close()

    lines: list[str] = text.splitlines()
    assert lines[0].startswith("Steps: 0 | Connected: 0/")
    assert set(lines[1]) == {"-"}
    assert len(lines[1]) >= len(lines[0])
    assert len(lines) == 2 + env.H


def test_render_returns_text_for_ansi_mode() -> None:
    """``render`` returns the textual board when the environment was created with the ansi render mode."""
    env = gym.make("NumberLinkRGB-v0", render_mode="ansi", level_id="builtin_7x7_ham_6c")
    try:
        env.reset(seed=1)
        frame = env.render()
    finally:
        env.close()

    assert isinstance(frame, str)
    assert "Steps:" in frame


def test_render_text_reports_bridge_lanes() -> None:
    """Bridge cells render two lane characters so both lanes stay visible in text mode."""
    env = NumberLinkRGBEnv(
        render_mode="ansi",
        generator=GeneratorConfig(mode="random_walk", width=6, height=6, colors=3, bridges_probability=1.0, seed=4),
        variant=VariantConfig(bridges_enabled=True),
    )
    try:
        env.reset(seed=4)
        text: str = env.render_text()
    finally:
        env.close()

    assert "*" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["levels"],
        ["levels", "--contains", "builtin"],
        ["register"],
        ["board", "--level-id", "builtin_5x5_rw_4c"],
        ["board", "--level-id", "builtin_5x5_rw_4c", "--apply-solution"],
        ["board", "--gen-mode", "hamiltonian", "--gen-width", "5", "--gen-height", "5", "--gen-colors", "3"],
    ],
)
def test_cli_commands_exit_successfully(argv: list[str]) -> None:
    """Every non-interactive CLI command completes without raising and reports success."""
    assert cli_main(argv) == 0


def test_cli_without_command_prints_help() -> None:
    """Invoking the CLI with no subcommand prints help and reports success."""
    assert cli_main([]) == 0


def test_cli_board_reads_grid_file(tmp_path: pathlib.Path) -> None:
    """The board command renders a level loaded from a grid file."""
    grid_path = tmp_path / "board.txt"
    grid_path.write_text("\n".join(LEVELS["builtin_5x5_rw_4c"].grid) + "\n", encoding="utf-8")
    assert cli_main(["board", "--grid-file", str(grid_path)]) == 0
