"""
Command line interface for the :mod:`numberlink` package.

This module provides a small command line utility that can register the environment, print textual representations of
puzzle boards, list bundled level identifiers, and open an interactive viewer. The primary entry points are
:func:`build_parser` for constructing the argument parser and :func:`main` for the console entry point.

The subcommand handlers are implemented as :func:`handle_register`, :func:`handle_viewer`, :func:`handle_board`, and
:func:`handle_levels`. Use :func:`load_grid_from_file` to load a grid from a filesystem path and :func:`load_bridges`
to parse an optional JSON list of bridge coordinates.

See also :class:`numberlink.config.VariantConfig` for variant options and :class:`numberlink.env.NumberLinkRGBEnv`
for environment details used by the viewer and board commands.
"""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .config import GeneratorConfig, RenderConfig, VariantConfig
from .env import NumberLinkRGBEnv
from .levels import LEVELS, load_level_from_file
from .registration import register_numberlink_v0
from .viewer import NumberLinkViewer

if TYPE_CHECKING:
    from argparse import Namespace, _SubParsersAction
    from collections.abc import Callable, Sequence

    from .env import RenderMode
    from .levels import Level
    from .types import ActType, Coord


def build_parser() -> ArgumentParser:
    """Build the top level command line argument parser.

    The returned :class:`argparse.ArgumentParser` is configured with the following subcommands and their handlers
    :func:`handle_register`, :func:`handle_viewer`, :func:`handle_board`, and :func:`handle_levels`.

    :return: Configured argument parser ready to parse command line arguments.
    :rtype: argparse.ArgumentParser
    """
    parser: ArgumentParser = ArgumentParser(
        prog="numberlink", description="Utilities for registering, inspecting, and playing NumberLink puzzles."
    )
    # should support --version and -v
    parser.add_argument(
        "--version", "-v", action="version", version=f"NumberLink v{__import__('numberlink').__version__}"
    )
    subparsers: _SubParsersAction[ArgumentParser] = parser.add_subparsers(dest="command", title="commands")

    register_parser: ArgumentParser = subparsers.add_parser(
        "register",
        help="Register the NumberLink environment with Gymnasium.",
        description="Register the NumberLink environment id so Gymnasium can instantiate it.",
    )
    register_parser.add_argument("--quiet", action="store_true", help="Suppress confirmation output.")
    register_parser.set_defaults(handler=handle_register)

    viewer_parser: ArgumentParser = subparsers.add_parser(
        "viewer",
        help="Launch the interactive pygame viewer.",
        description="Open the pygame viewer to play a NumberLink level interactively.",
    )
    viewer_parser.add_argument(
        "--level-id",
        default=None,
        help=(
            "Identifier of a built-in level template. When omitted along with --grid-file, a random level is generated."
        ),
    )
    viewer_parser.add_argument(
        "--grid-file", type=Path, help="Path to a text file containing a grid specification. One string row per line."
    )
    viewer_parser.add_argument(
        "--bridges-file", type=Path, help="Optional JSON file with a list of bridge coordinates [[row, col], ...]."
    )
    viewer_parser.add_argument(
        "--cell-size", type=int, default=64, help="Render size in pixels for each cell in the viewer window."
    )
    viewer_parser.add_argument("--seed", type=int, help="Seed for environment reset to obtain reproducible layouts.")
    viewer_parser.add_argument(
        "--step-limit", type=int, help="Override the default maximum number of steps before truncation."
    )
    viewer_parser.add_argument(
        "--render-mode",
        choices=["human", "ansi", "rgb_array", "none"],
        default="rgb_array",
        help="Render mode passed to the environment. Use 'none' to disable automatic renders.",
    )
    viewer_parser.add_argument(
        "--must-fill",
        action=BooleanOptionalAction,
        default=True,
        help="Require all cells to be filled before the puzzle is solved.",
    )
    viewer_parser.add_argument(
        "--allow-diagonal",
        action=BooleanOptionalAction,
        default=None,
        help="Permit diagonal moves for each color path.",
    )
    viewer_parser.add_argument(
        "--bridges-enabled",
        action=BooleanOptionalAction,
        default=None,
        help="Enable bridge cells that contain two independent lanes.",
    )
    viewer_parser.add_argument(
        "--cell-switch",
        action=BooleanOptionalAction,
        default=None,
        help="Activate cell switching mode where cell colors can be assigned directly.",
    )
    viewer_parser.add_argument(
        "--apply-solution",
        action="store_true",
        help="Apply the baked-in solution path before opening the viewer if one exists.",
    )
    # Generator-related options for random default
    viewer_parser.add_argument("--gen-mode", choices=["random_walk", "hamiltonian"], default="random_walk")
    viewer_parser.add_argument("--gen-width", type=int, default=8)
    viewer_parser.add_argument("--gen-height", type=int, default=8)
    viewer_parser.add_argument("--gen-colors", type=int, default=7)
    viewer_parser.add_argument("--gen-min-path", type=int, default=3)
    # Render configuration options
    viewer_parser.add_argument(
        "--endpoint-border-thickness", type=int, help="Border thickness for endpoints in pixels."
    )
    viewer_parser.add_argument("--connection-color-adjustment", type=int, help="Color adjustment for connections.")
    viewer_parser.add_argument("--render-height", type=int, help="Total render height in pixels.")
    viewer_parser.add_argument("--render-width", type=int, help="Total render width in pixels.")
    viewer_parser.add_argument("--gridline-thickness", type=int, help="Gridline thickness in pixels.")
    viewer_parser.add_argument(
        "--show-endpoint-numbers", action=BooleanOptionalAction, help="Display color index numbers on endpoints."
    )
    viewer_parser.add_argument("--number-font-border-thickness", type=int, help="Font border thickness for numbers.")
    viewer_parser.add_argument("--number-font-min-scale", type=int, help="Minimum font scale for endpoint labels.")
    viewer_parser.add_argument("--number-font-max-scale", type=int, help="Maximum font scale for endpoint labels.")
    viewer_parser.add_argument(
        "--help-overlay-font-border-thickness", type=int, help="Font border thickness for help overlay."
    )
    viewer_parser.add_argument(
        "--help-overlay-background-alpha", type=int, help="Alpha value for help overlay (0-255)."
    )
    viewer_parser.add_argument(
        "--active-head-highlight-thickness", type=int, help="Thickness for active head highlight."
    )
    viewer_parser.add_argument("--cursor-highlight-thickness", type=int, help="Thickness for cursor highlight.")
    viewer_parser.add_argument(
        "--print-text-in-human-mode", action=BooleanOptionalAction, help="Print text representation in human mode."
    )
    viewer_parser.set_defaults(handler=handle_viewer)

    board_parser: ArgumentParser = subparsers.add_parser(
        "board",
        help="Print a textual representation of a level.",
        description="Render a board in text mode, optionally applying the stored solution.",
    )
    board_parser.add_argument(
        "--level-id",
        default=None,
        help=(
            "Identifier of a built-in level template. When omitted along with --grid-file, a random level is generated."
        ),
    )
    board_parser.add_argument(
        "--grid-file", type=Path, help="Path to a text file containing a grid specification. One string row per line."
    )
    board_parser.add_argument(
        "--bridges-file", type=Path, help="Optional JSON file with a list of bridge coordinates [[row, col], ...]."
    )
    board_parser.add_argument("--seed", type=int, help="Seed for environment reset to obtain reproducible layouts.")
    board_parser.add_argument(
        "--apply-solution",
        action="store_true",
        help="Apply the baked-in solution and print the resulting board if available.",
    )
    board_parser.add_argument(
        "--must-fill",
        action=BooleanOptionalAction,
        default=True,
        help="Require all cells to be filled before the puzzle is solved.",
    )
    board_parser.add_argument(
        "--allow-diagonal",
        action=BooleanOptionalAction,
        default=None,
        help="Permit diagonal moves for the text rendering.",
    )
    board_parser.add_argument(
        "--bridges-enabled",
        action=BooleanOptionalAction,
        default=None,
        help="Enable bridge cells that contain two independent lanes.",
    )
    board_parser.add_argument(
        "--cell-switch",
        action=BooleanOptionalAction,
        default=None,
        help="Activate cell switching mode for the text rendering.",
    )
    # Generator options for random default
    board_parser.add_argument("--gen-mode", choices=["random_walk", "hamiltonian"], default="random_walk")
    board_parser.add_argument("--gen-width", type=int, default=8)
    board_parser.add_argument("--gen-height", type=int, default=8)
    board_parser.add_argument("--gen-colors", type=int, default=5)
    board_parser.add_argument("--gen-min-path", type=int, default=3)
    # Render configuration options
    board_parser.add_argument("--endpoint-border-thickness", type=int, help="Border thickness for endpoints in pixels.")
    board_parser.add_argument("--connection-color-adjustment", type=int, help="Color adjustment for connections.")
    board_parser.add_argument("--render-height", type=int, help="Total render height in pixels.")
    board_parser.add_argument("--render-width", type=int, help="Total render width in pixels.")
    board_parser.add_argument("--gridline-thickness", type=int, help="Gridline thickness in pixels.")
    board_parser.add_argument(
        "--show-endpoint-numbers", action=BooleanOptionalAction, help="Display color index numbers on endpoints."
    )
    board_parser.add_argument("--number-font-border-thickness", type=int, help="Font border thickness for numbers.")
    board_parser.add_argument("--number-font-min-scale", type=int, help="Minimum font scale for endpoint labels.")
    board_parser.add_argument("--number-font-max-scale", type=int, help="Maximum font scale for endpoint labels.")
    board_parser.add_argument(
        "--help-overlay-font-border-thickness", type=int, help="Font border thickness for help overlay."
    )
    board_parser.add_argument("--help-overlay-background-alpha", type=int, help="Alpha value for help overlay (0-255).")
    board_parser.add_argument(
        "--active-head-highlight-thickness", type=int, help="Thickness for active head highlight."
    )
    board_parser.add_argument("--cursor-highlight-thickness", type=int, help="Thickness for cursor highlight.")
    board_parser.add_argument(
        "--print-text-in-human-mode", action=BooleanOptionalAction, help="Print text representation in human mode."
    )
    board_parser.set_defaults(handler=handle_board)

    levels_parser: ArgumentParser = subparsers.add_parser(
        "levels",
        help="List bundled level identifiers.",
        description="List available level identifiers bundled with the package.",
    )
    levels_parser.add_argument("--contains", help="Filter level ids that contain the provided substring.")
    levels_parser.set_defaults(handler=handle_levels)

    return parser


def load_grid_from_file(path_obj: Path) -> Sequence[str]:
    """Load a grid specification from disk using :func:`numberlink.levels.load_level_from_file`.

    Return the list of row strings stored in the loaded :class:`numberlink.levels.Level`.

    :param path_obj: Filesystem path to a grid file.
    :return: List of row strings read from the file.
    :raises ValueError: When the file contains no non empty rows.
    """
    lvl: Level = load_level_from_file(path_obj)
    if not lvl.grid:
        raise ValueError(f"grid file {path_obj} did not contain any rows")
    return lvl.grid


def load_solution_for_grid(path_obj: Path) -> list[list[Coord]] | None:
    """Load an optional solution adjacent to a grid file.

    Use :func:`numberlink.levels.load_level_from_file` to load the grid and return the
    :py:attr:`numberlink.levels.Level.solution` value when present, ``None`` when no solution was given with the grid.

    :param path_obj: Filesystem path to the grid file.
    :return: Solution paths or ``None`` when not available.
    """
    lvl: Level = load_level_from_file(path_obj)
    return lvl.solution


def load_bridges(bridges_path: Path | None) -> Sequence[Coord] | None:
    """Load bridge coordinates from a JSON file.

    Return a sequence of ``(row, column)`` pairs. The JSON file must contain a top level list. Each entry must be a two
    element sequence representing a row and column index. Strings and byte sequences are rejected. The function converts
    numeric values to Python integers and returns ``None`` when ``bridges_path`` is ``None``.

    :param bridges_path: Path to a JSON file or ``None`` to indicate no bridges should be loaded.
    :return: Sequence of coordinate pairs or ``None`` when no file was provided.
    :rtype: collections.abc.Sequence[tuple[int, int]] or None
    :raises ValueError: If any JSON entry is not a two element sequence of numeric values.
    """
    if bridges_path is None:
        return None
    data: list[Coord] = json.loads(bridges_path.read_text(encoding="utf-8"))
    bridges: list[Coord] = []
    for entry in data:
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)) or len(entry) != 2:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("bridge entries must be two-element sequences")
        row_val = int(entry[0])
        col_val = int(entry[1])
        bridges.append((row_val, col_val))
    return bridges


def build_variant_from_args(
    default_variant: VariantConfig,
    must_fill: bool | None,
    allow_diagonal: bool | None,
    bridges_enabled: bool | None,
    cell_switch: bool | None,
) -> VariantConfig:
    """Create a :class:`numberlink.config.VariantConfig` from CLI overrides.

    For each override parameter, when the provided value is not ``None`` the returned variant uses that value. When the
    provided value is ``None`` the corresponding attribute from ``default_variant`` is preserved.

    :param default_variant: Default variant to use for attributes that are not overridden.
    :param must_fill: Override for the ``must_fill`` option when not ``None``.
    :param allow_diagonal: Override for the ``allow_diagonal`` option when not ``None``.
    :param bridges_enabled: Override for the ``bridges_enabled`` option when not ``None``.
    :param cell_switch: Override for the ``cell_switching_mode`` option when not ``None``.
    :return: New :class:`numberlink.config.VariantConfig` instance with the requested overrides applied.
    :rtype: numberlink.config.VariantConfig
    """
    return VariantConfig(
        must_fill=default_variant.must_fill if must_fill is None else must_fill,
        allow_diagonal=default_variant.allow_diagonal if allow_diagonal is None else allow_diagonal,
        bridges_enabled=default_variant.bridges_enabled if bridges_enabled is None else bridges_enabled,
        cell_switching_mode=default_variant.cell_switching_mode if cell_switch is None else cell_switch,
    )


def build_render_config_from_args(args: Namespace) -> RenderConfig | None:
    """Create a :class:`numberlink.config.RenderConfig` from CLI arguments when overrides are present.

    Return ``None`` when no render config arguments were provided, allowing the environment to use its default
    configuration. When any render config option is specified, build and return a new
    :class:`numberlink.config.RenderConfig` with overrides applied.

    :param args: Parsed CLI arguments containing optional render config fields.
    :return: New :class:`numberlink.config.RenderConfig` instance when overrides are present, otherwise ``None``.
    :rtype: RenderConfig or None
    """
    default = RenderConfig()
    overrides: dict[str, int | bool | None] = {}

    # Map args to config fields
    if args.endpoint_border_thickness is not None:
        overrides["endpoint_border_thickness"] = args.endpoint_border_thickness
    if args.connection_color_adjustment is not None:
        overrides["connection_color_adjustment"] = args.connection_color_adjustment
    if args.render_height is not None:
        overrides["render_height"] = args.render_height
    if args.render_width is not None:
        overrides["render_width"] = args.render_width
    if args.gridline_thickness is not None:
        overrides["gridline_thickness"] = args.gridline_thickness
    if args.show_endpoint_numbers is not None:
        overrides["show_endpoint_numbers"] = args.show_endpoint_numbers
    if args.number_font_border_thickness is not None:
        overrides["number_font_border_thickness"] = args.number_font_border_thickness
    if args.number_font_min_scale is not None:
        overrides["number_font_min_scale"] = args.number_font_min_scale
    if args.number_font_max_scale is not None:
        overrides["number_font_max_scale"] = args.number_font_max_scale
    if args.help_overlay_font_border_thickness is not None:
        overrides["help_overlay_font_border_thickness"] = args.help_overlay_font_border_thickness
    if args.help_overlay_background_alpha is not None:
        overrides["help_overlay_background_alpha"] = args.help_overlay_background_alpha
    if args.active_head_highlight_thickness is not None:
        overrides["active_head_highlight_thickness"] = args.active_head_highlight_thickness
    if args.cursor_highlight_thickness is not None:
        overrides["cursor_highlight_thickness"] = args.cursor_highlight_thickness
    if args.print_text_in_human_mode is not None:
        overrides["print_text_in_human_mode"] = args.print_text_in_human_mode

    if not overrides:
        return None

    # Build a RenderConfig explicitly with typed fields
    # Explicitly type and validate each override to satisfy the type checker
    endpoint_border_thickness: int = default.endpoint_border_thickness
    if "endpoint_border_thickness" in overrides:
        v: int | bool | None = overrides["endpoint_border_thickness"]
        if isinstance(v, int):
            endpoint_border_thickness = v

    connection_color_adjustment: int = default.connection_color_adjustment
    if "connection_color_adjustment" in overrides:
        v = overrides["connection_color_adjustment"]
        if isinstance(v, int):
            connection_color_adjustment = v

    render_height: int | None = default.render_height
    if "render_height" in overrides:
        v = overrides["render_height"]
        if isinstance(v, int):
            render_height = v

    render_width: int | None = default.render_width
    if "render_width" in overrides:
        v = overrides["render_width"]
        if isinstance(v, int):
            render_width = v

    gridline_thickness: int = default.gridline_thickness
    if "gridline_thickness" in overrides:
        v = overrides["gridline_thickness"]
        if isinstance(v, int):
            gridline_thickness = v

    show_endpoint_numbers: bool = default.show_endpoint_numbers
    if "show_endpoint_numbers" in overrides:
        v = overrides["show_endpoint_numbers"]
        if isinstance(v, bool):
            show_endpoint_numbers = v

    number_font_border_thickness: int = default.number_font_border_thickness
    if "number_font_border_thickness" in overrides:
        v = overrides["number_font_border_thickness"]
        if isinstance(v, int):
            number_font_border_thickness = v

    number_font_min_scale: int = default.number_font_min_scale
    if "number_font_min_scale" in overrides:
        v = overrides["number_font_min_scale"]
        if isinstance(v, int):
            number_font_min_scale = v

    number_font_max_scale: int | None = default.number_font_max_scale
    if "number_font_max_scale" in overrides:
        v = overrides["number_font_max_scale"]
        if isinstance(v, int):
            number_font_max_scale = v

    help_overlay_font_border_thickness: int = default.help_overlay_font_border_thickness
    if "help_overlay_font_border_thickness" in overrides:
        v = overrides["help_overlay_font_border_thickness"]
        if isinstance(v, int):
            help_overlay_font_border_thickness = v

    help_overlay_background_alpha: int = default.help_overlay_background_alpha
    if "help_overlay_background_alpha" in overrides:
        v = overrides["help_overlay_background_alpha"]
        if isinstance(v, int):
            help_overlay_background_alpha = v

    active_head_highlight_thickness: int = default.active_head_highlight_thickness
    if "active_head_highlight_thickness" in overrides:
        v = overrides["active_head_highlight_thickness"]
        if isinstance(v, int):
            active_head_highlight_thickness = v

    cursor_highlight_thickness: int = default.cursor_highlight_thickness
    if "cursor_highlight_thickness" in overrides:
        v = overrides["cursor_highlight_thickness"]
        if isinstance(v, int):
            cursor_highlight_thickness = v

    print_text_in_human_mode: bool = default.print_text_in_human_mode
    if "print_text_in_human_mode" in overrides:
        v = overrides["print_text_in_human_mode"]
        if isinstance(v, bool):
            print_text_in_human_mode = v

    return RenderConfig(
        endpoint_border_thickness=endpoint_border_thickness,
        endpoint_border_color=default.endpoint_border_color,
        connection_color_adjustment=connection_color_adjustment,
        render_height=render_height,
        render_width=render_width,
        grid_background_color=default.grid_background_color,
        gridline_color=default.gridline_color,
        gridline_thickness=gridline_thickness,
        show_endpoint_numbers=show_endpoint_numbers,
        number_font_color=default.number_font_color,
        number_font_border_color=default.number_font_border_color,
        number_font_border_thickness=number_font_border_thickness,
        number_font_min_scale=number_font_min_scale,
        number_font_max_scale=number_font_max_scale,
        help_overlay_font_color=default.help_overlay_font_color,
        help_overlay_font_border_color=default.help_overlay_font_border_color,
        help_overlay_font_border_thickness=help_overlay_font_border_thickness,
        help_overlay_background_color=default.help_overlay_background_color,
        help_overlay_background_alpha=help_overlay_background_alpha,
        active_head_highlight_color=default.active_head_highlight_color,
        active_head_highlight_thickness=active_head_highlight_thickness,
        cursor_highlight_thickness=cursor_highlight_thickness,
        cursor_endpoint_highlight_color=default.cursor_endpoint_highlight_color,
        print_text_in_human_mode=print_text_in_human_mode,
    )


def handle_register(args: Namespace) -> int:
    """Handle the ``register`` subcommand.

    This function calls :func:`numberlink.registration.register_numberlink_v0` to register the environment id with the
    Gymnasium registry. When the ``args.quiet`` flag is not set a brief confirmation message is emitted to standard
    output. The function returns an exit code suitable for a console entry point.

    :param args: Parsed arguments from :func:`build_parser` with an attribute ``quiet`` of type ``bool``.
    :return: Process exit code. ``0`` indicates success.
    :rtype: int
    """
    register_numberlink_v0()
    if not args.quiet:
        print('Registered Gymnasium id "NumberLinkRGB-v0".')
    return 0


def handle_viewer(args: Namespace) -> int:
    """Handle the ``viewer`` subcommand.

    Open an interactive viewer when appropriate. The function loads an optional grid using :func:`load_grid_from_file`
    and optional bridge coordinates using :func:`load_bridges`. It builds a :class:`numberlink.config.VariantConfig`
    from CLI overrides using :func:`build_variant_from_args`. It creates a :class:`numberlink.env.NumberLinkRGBEnv`
    instance for the requested configuration and performs an initial :meth:`numberlink.env.NumberLinkRGBEnv.reset`.

    When the ``--apply-solution`` flag is provided and a stored solution is available the solution is applied by issuing
    calls to :meth:`numberlink.env.NumberLinkRGBEnv.step`. When ``--render-mode`` is ``ansi`` the textual representation
    from :meth:`numberlink.env.NumberLinkRGBEnv._render_text` is printed. For other render modes a
    :class:`numberlink.viewer.NumberLinkViewer` is constructed and its :meth:`numberlink.viewer.NumberLinkViewer.loop`
    method is invoked.

    The environment is always closed by calling :meth:`numberlink.env.NumberLinkRGBEnv.close` in a ``finally`` block.

    :param args: Parsed arguments from :func:`build_parser`.
    :return: Process exit code. ``0`` indicates success.
    :rtype: int
    """
    grid: Sequence[str] | None = None
    gen: GeneratorConfig | None = None
    file_solution: list[list[Coord]] | None = None
    if args.grid_file:
        grid = load_grid_from_file(args.grid_file)
        file_solution = load_solution_for_grid(args.grid_file)
    elif not args.level_id:
        gen = GeneratorConfig(
            mode=args.gen_mode,
            width=args.gen_width,
            height=args.gen_height,
            colors=args.gen_colors,
            min_path_length=args.gen_min_path,
            seed=args.seed,
        )
    bridges: Sequence[Coord] | None = load_bridges(args.bridges_file)
    default_variant = VariantConfig()
    variant: VariantConfig = build_variant_from_args(
        default_variant, args.must_fill, args.allow_diagonal, args.bridges_enabled, args.cell_switch
    )
    render_config: RenderConfig | None = build_render_config_from_args(args)
    render_mode: RenderMode | None = None if args.render_mode == "none" else args.render_mode
    env: NumberLinkRGBEnv = NumberLinkRGBEnv(
        grid=grid,
        render_mode=render_mode,
        level_id=None if grid or gen else args.level_id,
        variant=variant,
        bridges=bridges,
        generator=gen,
        step_limit=args.step_limit,
        render_config=render_config,
        solution=file_solution,
    )
    try:
        _, info = env.reset(seed=args.seed)
        # Status line
        connected_count: int = env._closed.sum()
        status: str = (
            f"Viewer: {env.W}x{env.H} | colors={env.num_colors} | "
            f"connected={connected_count}/{env.num_colors} | steps={info.get('steps', 0)}"
        )
        print(status)
        if args.apply_solution:
            solution: list[ActType] | None = env.get_solution()
            if solution:
                for action in solution:
                    env.step(action)
            else:
                print("No stored solution available for this configuration.")
        if args.render_mode == "ansi":
            print(env._render_text())
        else:
            viewer: NumberLinkViewer = NumberLinkViewer(env, cell_size=args.cell_size)
            viewer.loop()
    finally:
        env.close()
    return 0


def handle_board(args: Namespace) -> int:
    """Handle the ``board`` subcommand.

    Print a textual rendering of a requested level. The function behaves similarly to :func:`handle_viewer` for argument
    parsing and variant construction. It creates a :class:`numberlink.env.NumberLinkRGBEnv` with ``render_mode`` set to
    ``ansi`` and prints the textual board returned by :meth:`numberlink.env.NumberLinkRGBEnv._render_text` after calling
    :meth:`numberlink.env.NumberLinkRGBEnv.reset`.

    When the ``--apply-solution`` flag is provided the function attempts to retrieve a stored solution via
    :meth:`numberlink.env.NumberLinkRGBEnv.get_solution`. If a solution is found it is applied stepwise using
    :meth:`numberlink.env.NumberLinkRGBEnv.step` and the resulting board is printed.

    The environment is closed by calling :meth:`numberlink.env.NumberLinkRGBEnv.close` in a ``finally`` block.

    :param args: Parsed arguments from :func:`build_parser`.
    :return: Process exit code. ``0`` indicates success.
    :rtype: int
    """
    grid: Sequence[str] | None = None
    gen: GeneratorConfig | None = None
    file_solution: list[list[Coord]] | None = None
    if args.grid_file:
        grid = load_grid_from_file(args.grid_file)
        file_solution = load_solution_for_grid(args.grid_file)
    elif not args.level_id:
        gen = GeneratorConfig(
            mode=args.gen_mode,
            width=args.gen_width,
            height=args.gen_height,
            colors=args.gen_colors,
            min_path_length=args.gen_min_path,
            seed=args.seed,
        )
    bridges: Sequence[Coord] | None = load_bridges(args.bridges_file)
    default_variant: VariantConfig = VariantConfig()
    variant: VariantConfig = build_variant_from_args(
        default_variant, args.must_fill, args.allow_diagonal, args.bridges_enabled, args.cell_switch
    )
    render_config: RenderConfig | None = build_render_config_from_args(args)
    env: NumberLinkRGBEnv = NumberLinkRGBEnv(
        grid=grid,
        render_mode="ansi",
        level_id=None if grid or gen else args.level_id,
        variant=variant,
        bridges=bridges,
        generator=gen,
        render_config=render_config,
        solution=file_solution,
    )
    try:
        _, info = env.reset(seed=args.seed)
        print(env._render_text())
        connected_count: int = env._closed.sum()
        status: str = (
            f"Status: {env.W}x{env.H} | colors={env.num_colors} | "
            f"connected={connected_count}/{env.num_colors} | steps={info.get('steps', 0)}"
        )
        print(status)
        if args.apply_solution:
            solution: list[ActType] | None = env.get_solution()
            if not solution:
                print("No stored solution found for this configuration.")
            else:
                for action in solution:
                    env.step(action)
                print()
                print("After applying solution:")
                print(env._render_text())
    finally:
        env.close()
    return 0


def handle_levels(args: Namespace) -> int:
    """Handle the ``levels`` subcommand and print bundled level identifiers.

    The function enumerates the keys of :data:`numberlink.levels.LEVELS` and optionally filters them by substring when
    ``args.contains`` is set.

    :param args: Parsed arguments from :func:`build_parser` with an optional attribute ``contains`` used for
        substring filtering.
    :return: Process exit code. ``0`` indicates success.
    :rtype: int
    """
    ids: list[str] = sorted(LEVELS.keys())
    if args.contains:
        ids = [level_id for level_id in ids if args.contains in level_id]
    if ids:
        print(f"Available levels ({len(ids)}):")
        for level_id in ids:
            print(f"  - {level_id}")
    else:
        print("No levels matched the filter.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point for the package.

    The function constructs the parser via :func:`build_parser`, parses the provided argument list, and dispatches to
    the handler bound on the chosen subparser. When no subcommand is provided the parser help message is printed and
    the function returns ``0``.

    :param argv: Optional sequence of arguments to parse. When ``None`` the system argv is used by :mod:`argparse`.
    :return: Process exit code produced by the selected subcommand handler or ``0`` when no command was requested.
    :rtype: int
    """
    parser: ArgumentParser = build_parser()
    args: Namespace = parser.parse_args(argv)
    handler: Callable[[Namespace], int] | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)
