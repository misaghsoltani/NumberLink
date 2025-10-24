"""Interactive pygame viewer for :class:`numberlink.env.NumberLinkRGBEnv`.

Provide an interactive display and input handling for a running :class:`numberlink.env.NumberLinkRGBEnv` instance.
Open a :mod:`pygame` window that renders environment state and accepts keyboard input to manipulate the environment in
either path editing mode or cell switching mode.

The primary entry point is :class:`numberlink.viewer.NumberLinkViewer`.
"""

from __future__ import annotations

from collections import OrderedDict
from importlib import import_module
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast
import warnings

import numpy as np

from .env import NumberLinkRGBEnv
from .number_render import BITMAP_FONT, build_endpoint_labels

if TYPE_CHECKING:
    from types import ModuleType

    import gymnasium as gym
    from IPython.core.interactiveshell import InteractiveShell
    from numpy.typing import NDArray
    from pygame import Rect, Surface
    from pygame.time import Clock

    from .types import ActType, CellLane, Coord, RGBInt, Snapshot


GLYPH_CACHE_MAX: int = 512


class NumberLinkViewer:
    """Interactive viewer for a :class:`numberlink.env.NumberLinkRGBEnv` instance.

    Forward keyboard input to the environment. Support two interaction modes.
    In path editing mode, arrow keys and optional diagonal keys extend or retract a selected path head by sending
    directional actions to :meth:`numberlink.env.NumberLinkRGBEnv.step`. In cell switching mode, a cursor moves across
    the board and the highlighted cell can be painted using an action created by
    :meth:`numberlink.env.NumberLinkRGBEnv.encode_cell_switching_action`.

    Controls

    ``Path mode``

    - Move the selected head with arrow or diagonal keys.
    - Switch selected head with the left and right bracket keys.
    - Cycle the active color with Tab and Shift+Tab.
    - Backtrack the active head by one step using :meth:`_backtrack_selected`.

    ``Cell mode``

    - Move the cursor with arrow or diagonal keys.
    - Cycle the active color with Tab and Shift+Tab.
    - Select a color using digit keys when available.
    - Paint the highlighted cell using :meth:`_paint_selected_cell`. The action is forwarded to
      :meth:`numberlink.env.NumberLinkRGBEnv.step`.

    :param env: Environment to render and control. If ``env`` is wrapped the underlying
        :class:`numberlink.env.NumberLinkRGBEnv` instance is obtained from ``env.unwrapped``.
    :type env: :class:`numberlink.env.NumberLinkRGBEnv` or wrapper
    :param cell_size: Pixel size of each rendered cell. Defaults to ``48``.
    :type cell_size: int, optional

    :ivar env: The unwrapped :class:`numberlink.env.NumberLinkRGBEnv` instance used for rendering and action dispatch.
    :vartype env: :class:`numberlink.env.NumberLinkRGBEnv`
    :ivar cell: Pixel size of each rendered cell.
    :vartype cell: int
    :ivar sel_color: Index of the currently selected color.
    :vartype sel_color: int
    :ivar sel_head: Index of the currently selected head for the active color.
    :vartype sel_head: int
    :ivar switch_mode: Whether the viewer is operating in cell switching mode.
    :vartype switch_mode: bool
    :ivar cursor: Row and column of the cursor when in cell switching mode.
    :vartype cursor: list[int]
    :ivar window: The active :mod:`pygame` window Surface or ``None`` if no window is open.
    :vartype window: pygame.Surface or None
    :ivar pygame: Reference to the loaded :mod:`pygame` module after initialization.
    :vartype pygame: module or None

    .. seealso::
       :mod:`pygame` for window and input primitives used by the viewer.

    .. note::
       The viewer accesses internal environment attributes, e.g., :attr:`numberlink.env.NumberLinkRGBEnv._endpoint_mask`
       and :attr:`numberlink.env.NumberLinkRGBEnv._palette_stack` for efficient rendering.
    """

    _glyph_cache: OrderedDict[tuple[str, int, int, RGBInt, RGBInt | None], Surface]
    HELP_AUTO_HIDE_MS: int = 4000

    def __init__(self, env: NumberLinkRGBEnv | gym.Env[NDArray[np.uint8], np.int64], cell_size: int = 48) -> None:
        """Initialize the viewer and internal state for a given environment.

        Unwrap ``env`` when it is a wrapped environment, initialize the selected color and head indices, and position
        the cursor when in cell-switching mode.

        :param env: The environment to render and control. If ``env`` is wrapped, the underlying environment is obtained
            from ``env.unwrapped``.
        :type env: :class:`numberlink.env.NumberLinkRGBEnv` or wrapper
        :param cell_size: Pixel size of each rendered cell. Defaults to ``48``.
        :type cell_size: int, optional
        :raises ValueError: If ``cell_size`` is less than or equal to zero.
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")

        import warnings  # noqa: PLC0415

        warnings.filterwarnings(
            action="ignore",
            message=r".*pkg_resources is deprecated as an API.*",
            category=UserWarning,
            module=r"pygame\.pkgdata",
        )

        self.pygame: ModuleType = import_module("pygame")
        self._pygame_initialized: bool = False

        # Accept wrapped envs from gym.make and unwrap to the base env for direct state access
        self.env: NumberLinkRGBEnv = cast(NumberLinkRGBEnv, env.unwrapped)
        self.cell: int = cell_size
        self.sel_color: int = 0
        self.sel_head: int = 0
        self.switch_mode: bool = self.env.variant.cell_switching_mode
        self.cursor: list[int] = [0, 0]
        if self.switch_mode:
            free_cells: NDArray[np.int_] = np.argwhere(~self.env._endpoint_mask)
            if free_cells.size > 0:
                first: NDArray[np.int_] = free_cells[0]
                self.cursor = [int(first[0]), int(first[1])]
            # If no free cells exist (entire grid is endpoints), cursor remains at [0, 0]

        self.window: Surface | None = None
        self.show_help: bool = True
        self._help_auto_hide_deadline: int | None = None
        self.mouse_dragging: bool = False
        self.last_mouse_cell: Coord | None = None

        # Solution replay state
        self._replay_state: str = "idle"  # one of: 'idle', 'playing', 'paused'
        self._replay_solution: list[ActType] | None = None
        self._replay_index: int = 0
        self._replay_interval_ms: int = 150
        self._replay_last_advance_ms: int = 0
        self._pre_replay_snapshot: Snapshot | None = None
        self._replay_info_message: str | None = None
        self._replay_info_deadline_ms: int | None = None

        # Cache for pre-rendered glyph surfaces keyed by (char, scale, outline_thickness, fg, outline)
        self._glyph_cache: OrderedDict[tuple[str, int, int, RGBInt, RGBInt | None], Surface] = OrderedDict()

    def _initialize_pygame(self) -> None:
        """Initialize pygame once per viewer instance and set the application icon."""
        if self._pygame_initialized:
            return

        pygame: ModuleType = self.pygame
        pygame.init()
        try:
            png_path: Path = (Path(__file__).parent / "assets" / "numberlink-logo.png").resolve()
            if png_path.exists():
                icon_surf: Surface = pygame.image.load(str(png_path))
                pygame.display.set_icon(icon_surf)
        except Exception:
            pass

        if self.show_help:
            self._help_auto_hide_deadline = pygame.time.get_ticks() + self.HELP_AUTO_HIDE_MS
        else:
            self._help_auto_hide_deadline = None
        self._pygame_initialized = True

    def _shutdown_pygame(self) -> None:
        """Shut down pygame when the viewer loop exits."""
        if not self._pygame_initialized:
            return
        try:
            self.pygame.quit()
        finally:
            self._pygame_initialized = False
            self.window = None

    def loop(self) -> None:
        """Run the interactive :mod:`pygame` event loop.

        Load :mod:`pygame` on first use, create a window sized to the environment grid, and process input events until
        the user exits.
        """
        if _try_launch_notebook_viewer(self):
            return

        self._initialize_pygame()
        # Validate environment dimensions
        if self.env.W <= 0 or self.env.H <= 0:
            raise ValueError(f"Invalid environment dimensions: W={self.env.W}, H={self.env.H}")

        try:
            pygame: ModuleType = self.pygame
            self.window = pygame.display.set_mode((self.env.W * self.cell, self.env.H * self.cell))

            clock: Clock = pygame.time.Clock()
            running: bool = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        # Global controls available in any state
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_h:
                            self._toggle_help_overlay()
                        elif event.key == pygame.K_r:
                            # Reset cancels any active replay
                            self._stop_replay(restore=False)
                            self._reset_environment()
                        elif event.key == pygame.K_n:
                            # New level cancels any active replay
                            self._stop_replay(restore=False)
                            self._generate_new_level()
                        elif event.key == pygame.K_t:
                            # Toggle solution replay play/pause or start
                            self._toggle_replay()
                        elif event.key == pygame.K_s:
                            # Stop and restore pre-replay state
                            self._stop_replay(restore=True)
                        elif self._replay_state in {"playing", "paused"}:
                            # While replay is active, ignore other interactive inputs
                            pass
                        elif self.switch_mode:
                            self._handle_cell_switch_key(event.key)
                        else:
                            self._handle_path_key(event.key)
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1 and self._replay_state == "idle":  # Left click
                            self._handle_mouse_down(event.pos)
                    elif event.type == pygame.MOUSEMOTION:
                        if self.mouse_dragging and self._replay_state == "idle":
                            self._handle_mouse_motion(event.pos)
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.mouse_dragging = False
                        self.last_mouse_cell = None

                mode_label: str = "Cell mode" if self.switch_mode else "Path mode"
                caption: str = (
                    f"NumberLinkRGB-v0 Viewer | {mode_label} | Color {self.sel_color}/{self.env.num_colors - 1}"
                )
                if not self.switch_mode:
                    caption += f" | Head {self.sel_head}"
                pygame.display.set_caption(caption)

                self._update_help_timer()
                self._advance_replay_if_needed()
                self._draw()
                pygame.display.flip()
                clock.tick(30)
        finally:
            self._shutdown_pygame()

    def _handle_path_key(self, key: int) -> None:
        """Handle a single key event while in path-editing mode.

        Map navigation and control keys to environment actions. Direction keys are translated to a direction index with
        :meth:`_direction_index` and the resulting action is dispatched via :meth:`_attempt_path_step` which calls
        :meth:`numberlink.env.NumberLinkRGBEnv.step`.

        :param key: Numeric key code from :mod:`pygame`.
        :type key: int
        """
        pygame: ModuleType = self.pygame
        mods: int = pygame.key.get_mods()
        if key == pygame.K_TAB and (mods & pygame.KMOD_SHIFT):
            self._cycle_color(-1)
            return
        if key == pygame.K_TAB:
            self._cycle_color(1)
            return
        if key == pygame.K_LEFTBRACKET:
            self.sel_head = 0
            return
        if key == pygame.K_RIGHTBRACKET:
            self.sel_head = 1
            return
        if key == pygame.K_SPACE:
            self._backtrack_selected()
            return

        move_map: dict[int, Coord] = {
            pygame.K_UP: (-1, 0),
            pygame.K_RIGHT: (0, 1),
            pygame.K_DOWN: (1, 0),
            pygame.K_LEFT: (0, -1),
            pygame.K_q: (-1, -1),
            pygame.K_e: (-1, 1),
            pygame.K_z: (1, -1),
            pygame.K_c: (1, 1),
        }
        dr: int
        dc: int
        if key in move_map:
            dr, dc = move_map[key]
            self._attempt_path_step(dr, dc)

    def _handle_cell_switch_key(self, key: int) -> None:
        """Handle a single key event while in cell-switching mode.

        Update the cursor location, change the selected color, or paint the highlighted cell. Painting uses
        :meth:`numberlink.env.NumberLinkRGBEnv.encode_cell_switching_action` to create an action which is forwarded to
        :meth:`numberlink.env.NumberLinkRGBEnv.step`.

        :param key: Numeric key code from :mod:`pygame`.
        :type key: int
        """
        pygame: ModuleType = self.pygame
        mods: int = pygame.key.get_mods()
        if key == pygame.K_TAB and (mods & pygame.KMOD_SHIFT):
            self._cycle_color(-1)
            return

        if key == pygame.K_TAB:
            self._cycle_color(1)
            return

        digit_map: dict[int, int] = {
            pygame.K_0: -1,  # 0 means clear cell (color value 0)
            pygame.K_1: 0,
            pygame.K_2: 1,
            pygame.K_3: 2,
            pygame.K_4: 3,
            pygame.K_5: 4,
            pygame.K_6: 5,
            pygame.K_7: 6,
            pygame.K_8: 7,
            pygame.K_9: 8,
        }
        # Support numpad digits as well
        digit_map.update({
            pygame.K_KP_0: -1,
            pygame.K_KP_1: 0,
            pygame.K_KP_2: 1,
            pygame.K_KP_3: 2,
            pygame.K_KP_4: 3,
            pygame.K_KP_5: 4,
            pygame.K_KP_6: 5,
            pygame.K_KP_7: 6,
            pygame.K_KP_8: 7,
            pygame.K_KP_9: 8,
        })
        if key in digit_map:
            idx: int = digit_map[key]
            if idx == -1:
                # Clear the cell
                row, col = self.cursor
                if 0 <= row < self.env.H and 0 <= col < self.env.W and not self.env._endpoint_mask[row, col]:
                    # Only attempt to clear if cell is not already empty
                    cell_occupied: bool = (
                        self.env._grid_codes[row, col] != 0
                        or self.env._lane_v[row, col] != 0
                        or self.env._lane_h[row, col] != 0
                    )
                    if cell_occupied:
                        action: int = self.env.encode_cell_switching_action(row, col, 0)
                        self.env.step(action=action)
            elif idx < self.env.num_colors:
                self.sel_color = idx
            return

        move_map: dict[int, Coord] = {
            pygame.K_UP: (-1, 0),
            pygame.K_RIGHT: (0, 1),
            pygame.K_DOWN: (1, 0),
            pygame.K_LEFT: (0, -1),
            pygame.K_q: (-1, -1),
            pygame.K_e: (-1, 1),
            pygame.K_z: (1, -1),
            pygame.K_c: (1, 1),
        }
        dr: int
        dc: int
        if key in move_map:
            dr, dc = move_map[key]
            self._move_cursor(dr, dc)
            return

        if key in {pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER}:
            self._paint_selected_cell()

    def _toggle_help_overlay(self) -> None:
        """Toggle the help overlay and reset the auto-hide timer."""
        self.show_help = not self.show_help
        self._help_auto_hide_deadline = (
            self.pygame.time.get_ticks() + self.HELP_AUTO_HIDE_MS if self.show_help else None
        )

    def _reset_environment(self, seed: int | None = None) -> None:
        """Reset the environment and refresh viewer state."""
        self.env.reset(seed=seed)
        self._reset_view_state()

    def _generate_new_level(self) -> None:
        """Generate a new level when a generator configuration is available."""
        if not self._can_generate_new_level():
            return
        try:
            self.env.regenerate_level()
        except Exception:  # Silently ignore any generation errors and keep current level
            return
        self._reset_view_state()

    # Solution replay helpers

    def _toggle_replay(self) -> None:
        """Start replay when idle and toggle play or pause when already initialized.

        When the replay state is ``idle`` call :meth:`_start_replay`. When the replay state is ``playing`` switch to
        ``paused``. When the replay state is ``paused`` resume playback and update the timing cursor.
        """
        if self._replay_state == "idle":
            self._start_replay()
            return
        if self._replay_state == "playing":
            self._replay_state = "paused"
            return
        if self._replay_state == "paused":
            self._replay_state = "playing"
            self._replay_last_advance_ms = self.pygame.time.get_ticks()

    def _start_replay(self) -> None:
        """Begin solution replay from the level's start state if a solution is available."""
        # Obtain solution from environment
        solution: list[ActType] | None = self.env.get_solution()
        if not solution:
            # Show a brief message to the user
            self._replay_info_message = "No solution available for this level"
            self._replay_info_deadline_ms = self.pygame.time.get_ticks() + 2500
            return

        # Snapshot current env and viewer state for restoration
        self._pre_replay_snapshot = self._snapshot_state()
        # Reset to the exact start state
        self.env.reset()
        self._reset_view_state()
        # Initialize replay cursor
        self._replay_solution = solution
        self._replay_index = 0
        self._replay_last_advance_ms = self.pygame.time.get_ticks()
        self._replay_state = "playing"

    def _stop_replay(self, *, restore: bool) -> None:
        """Stop any active replay and optionally restore the pre-replay environment state."""
        if self._replay_state == "idle":
            self._pre_replay_snapshot = None
            return
        self._replay_state = "idle"
        self._replay_solution = None
        self._replay_index = 0
        self._replay_last_advance_ms = 0
        if restore and self._pre_replay_snapshot is not None:
            self._restore_state(self._pre_replay_snapshot)
            self._pre_replay_snapshot = None
            self._reset_view_state()
        # Clear transient info message
        self._replay_info_message = None
        self._replay_info_deadline_ms = None

    def _advance_replay_if_needed(self) -> None:
        """Advance the solution replay based on the configured interval."""
        if self._replay_state != "playing" or self._replay_solution is None:
            # Expire transient message timer if any
            if (
                self._replay_info_deadline_ms is not None
                and self.pygame.time.get_ticks() >= self._replay_info_deadline_ms
            ):
                self._replay_info_message = None
                self._replay_info_deadline_ms = None
            return

        now: int = self.pygame.time.get_ticks()
        if now - self._replay_last_advance_ms < self._replay_interval_ms:
            return
        self._replay_last_advance_ms = now

        if self._replay_index >= len(self._replay_solution):
            # Finished, leave solved state and re-enable all inputs
            self._replay_state = "idle"
            self._replay_solution = None
            return

        action: int = int(self._replay_solution[self._replay_index])
        self.env.step(action=action)
        self._replay_index += 1

    def _snapshot_state(self) -> Snapshot:
        """Capture a deep copy of environment and viewer state for later restoration."""
        snap: Snapshot = {
            "_grid_codes": self.env._grid_codes.copy(),
            "_lane_v": self.env._lane_v.copy(),
            "_lane_h": self.env._lane_h.copy(),
            # Heads and stacks: capture the runtime tuple structures
            "_heads": [[(r, c) for (r, c) in heads] for heads in self.env._heads],
            "_stacks": [[[(r, c, lane) for (r, c, lane) in arm] for arm in color] for color in self.env._stacks],
            "_closed": self.env._closed.copy(),
            "_steps": self.env._steps,
            # Viewer selections
            "sel_color": self.sel_color,
            "sel_head": self.sel_head,
            "switch_mode": self.switch_mode,
            "cursor": [self.cursor[0], self.cursor[1]],
        }
        return snap

    def _restore_state(self, snap: Snapshot) -> None:
        """Restore environment and viewer state from a snapshot obtained via :meth:`_snapshot_state`."""
        # Environment arrays
        self.env._grid_codes[:, :] = snap["_grid_codes"]
        self.env._lane_v[:, :] = snap["_lane_v"]
        self.env._lane_h[:, :] = snap["_lane_h"]
        self.env._heads = [[(r, c) for (r, c) in heads] for heads in snap["_heads"]]
        # Rebuild stacks with tuples
        self.env._stacks = [[[(r, c, lane) for (r, c, lane) in arm] for arm in color] for color in snap["_stacks"]]
        self.env._closed[:] = snap["_closed"]
        self.env._steps = snap["_steps"]
        # Viewer selections
        self.sel_color = snap["sel_color"]
        self.sel_head = snap["sel_head"]
        self.switch_mode = bool(snap["switch_mode"])

        self.cursor = [snap["cursor"][0], snap["cursor"][1]]

    def _reset_view_state(self) -> None:
        """Synchronize viewer selections and cursor with the environment."""
        self.sel_color = max(0, self.env.num_colors - 1) if self.env.num_colors > 0 else 0
        self.sel_head = 0
        self.switch_mode = self.env.variant.cell_switching_mode
        if self.switch_mode:
            free_cells: NDArray[np.int_] = np.argwhere(~self.env._endpoint_mask)
            if free_cells.size > 0:
                first: NDArray[np.int_] = free_cells[0]
                self.cursor = [int(first[0]), int(first[1])]
            else:
                self.cursor = [0, 0]
        else:
            self.cursor = [0, 0]
        self.mouse_dragging = False
        self.last_mouse_cell = None

    def _can_generate_new_level(self) -> bool:
        """Return whether the viewer can request a new procedurally generated level."""
        return hasattr(self.env, "regenerate_level") and getattr(self.env, "_generator_config", None) is not None

    def _update_help_timer(self) -> None:
        """Hide the help overlay automatically after the configured timeout."""
        if not self.show_help:
            self._help_auto_hide_deadline = None
            return
        if self._help_auto_hide_deadline is None:
            return
        if self.pygame.time.get_ticks() >= self._help_auto_hide_deadline:
            self.show_help = False
            self._help_auto_hide_deadline = None

    def _cycle_color(self, delta: int) -> None:
        """Cycle the currently selected color by ``delta``.

        Wrap the selection using the environment's ``num_colors`` attribute.

        :param delta: Signed amount to change the selected color index.
        :type delta: int
        """
        if self.env.num_colors <= 0:
            return
        self.sel_color = (self.sel_color + delta) % self.env.num_colors

    def _direction_index(self, dr: int, dc: int) -> int | None:
        """Return the index of the direction matching the delta ``(dr, dc)``.

        Search :attr:`numberlink.env.NumberLinkRGBEnv._dirs` for an entry whose row and column deltas match ``dr`` and
        ``dc``. Return the matching index or ``None`` when no match exists.

        :param dr: Row delta.
        :type dr: int
        :param dc: Column delta.
        :type dc: int
        :return: Direction index or ``None``.
        :rtype: int or None
        """
        for k in range(self.env._num_dirs):
            if self.env._dirs[k][0] == dr and self.env._dirs[k][1] == dc:
                return k

        return None

    def _attempt_path_step(self, dr: int, dc: int) -> None:
        """Attempt to move the selected head in direction ``(dr, dc)``.

        Convert the direction to an index using :meth:`_direction_index`. If a valid index exists compute the action
        index for the selected color and head and dispatch it via :meth:`numberlink.env.NumberLinkRGBEnv.step`.

        :param dr: Row delta for the attempted step.
        :type dr: int
        :param dc: Column delta for the attempted step.
        :type dc: int
        """
        d_index: int | None = self._direction_index(dr, dc)
        if d_index is None or self.sel_color < 0 or self.sel_color >= self.env.num_colors:
            return
        self._ensure_color_ready_for_head()
        base: int = self.sel_color * self.env._actions_per_color + self.sel_head * self.env._num_dirs
        self.env.step(action=base + d_index)

    def _backtrack_selected(self) -> None:
        """Backtrack the active head one step when possible.

        If the active head has at least two entries on its stack, compute the reverse step to return the head to the
        previous cell and dispatch the corresponding action using :meth:`numberlink.env.NumberLinkRGBEnv.step`.
        """
        if self.sel_color < 0 or self.sel_color >= self.env.num_colors:
            return
        stacks: list[CellLane] = self.env._stacks[self.sel_color][self.sel_head]
        if len(stacks) < 2:
            return
        hr, hc = self.env._heads[self.sel_color][self.sel_head]
        pr, pc, _lane = stacks[-2]
        dr, dc = pr - hr, pc - hc
        d_index: int | None = self._direction_index(dr, dc)
        if d_index is None:
            return
        base: int = self.sel_color * self.env._actions_per_color + self.sel_head * self.env._num_dirs
        self.env.step(action=base + d_index)

    def _move_cursor(self, dr: int, dc: int) -> None:
        """Move the cursor in cell-switching mode by ``(dr, dc)``.

        Clamp the cursor coordinates to the valid grid range.

        :param dr: Row delta.
        :type dr: int
        :param dc: Column delta.
        :type dc: int
        """
        row: int = max(0, min(self.env.H - 1, self.cursor[0] + dr))
        col: int = max(0, min(self.env.W - 1, self.cursor[1] + dc))
        self.cursor = [row, col]

    def _paint_selected_cell(self) -> None:
        """Paint the highlighted cell with the currently selected color.

        Do nothing for endpoint cells. For other cells, construct the action using
        :meth:`numberlink.env.NumberLinkRGBEnv.encode_cell_switching_action` and dispatch it to
        :meth:`numberlink.env.NumberLinkRGBEnv.step`.
        """
        row, col = self.cursor
        if (
            row < 0
            or row >= self.env.H
            or col < 0
            or col >= self.env.W
            or self.env._endpoint_mask[row, col]
            or self.sel_color < 0
            or self.sel_color >= self.env.num_colors
        ):
            return

        # Skip if the cell already has the selected color value
        desired_code: int = self.sel_color + 1
        if not self.env._bridges[row, col]:
            if int(self.env._grid_codes[row, col]) == desired_code:
                return
        elif int(self.env._lane_v[row, col]) == desired_code and int(self.env._lane_h[row, col]) == desired_code:
            return

        action: int = self.env.encode_cell_switching_action(row, col, desired_code)
        self.env.step(action=action)

    def _handle_mouse_down(self, pos: Coord) -> None:
        """Handle mouse button down event.

        Convert pixel position to grid coordinates and initiate dragging.

        :param pos: Mouse position in pixels as (x, y).
        :type pos: Coord
        """
        if len(pos) < 2 or self.cell <= 0:
            return

        col: int = pos[0] // self.cell
        row: int = pos[1] // self.cell
        if 0 <= row < self.env.H and 0 <= col < self.env.W:
            self.mouse_dragging = True
            self.last_mouse_cell = (row, col)
            if self.switch_mode:
                self.cursor = [row, col]
                self._paint_selected_cell()
            else:
                # In path mode, find which head to move or select
                self._handle_mouse_cell(row, col)

    def _handle_mouse_motion(self, pos: Coord) -> None:
        """Handle mouse motion during drag.

        Convert pixel position to grid coordinates and perform action if cell changed.

        :param pos: Mouse position in pixels as (x, y).
        :type pos: Coord
        """
        if len(pos) < 2 or self.cell <= 0:
            return

        col: int = pos[0] // self.cell
        row: int = pos[1] // self.cell
        if 0 <= row < self.env.H and 0 <= col < self.env.W:
            current_cell: Coord = (row, col)
            if self.last_mouse_cell != current_cell:
                self.last_mouse_cell = current_cell
                if self.switch_mode:
                    self.cursor = [row, col]
                    self._paint_selected_cell()
                else:
                    self._handle_mouse_cell(row, col)

    def _select_focus_for_cell(self, row: int, col: int) -> bool:
        """Adjust the selected color and head based on the cell at ``(row, col)``.

        Prioritize exact matches against endpoints and current head positions. When the cell belongs to a path segment,
        pick the head whose stack approaches the cell most recently. Bridge cells prefer the lane that matches the
        currently selected color before falling back to any occupied lane.

        :param row: Cell row index.
        :type row: int
        :param col: Cell column index.
        :type col: int
        :return: ``True`` when the selection changed, otherwise ``False``.
        :rtype: bool
        """
        if row < 0 or row >= self.env.H or col < 0 or col >= self.env.W or self.env.num_colors <= 0:
            return False

        initial_color: int = self.sel_color
        initial_head: int = self.sel_head

        target_color: int | None = None
        target_head: int | None = None

        # Endpoints take precedence
        for ci, endpoints in enumerate(self.env._endpoints):
            if endpoints[0] == (row, col):
                target_color = ci
                target_head = 0
                break
            if endpoints[1] == (row, col):
                target_color = ci
                target_head = 1
                break

        # Current head locations
        if target_color is None:
            for ci in range(self.env.num_colors):
                head0_r, head0_c = self.env._heads[ci][0]
                if head0_r == row and head0_c == col:
                    target_color = ci
                    target_head = 0
                    break
                head1_r, head1_c = self.env._heads[ci][1]
                if head1_r == row and head1_c == col:
                    target_color = ci
                    target_head = 1
                    break

        # Stacks (path segments already traced)
        if target_color is None:
            for ci in range(self.env.num_colors):
                stacks: list[list[CellLane]] = self.env._stacks[ci]
                chosen_head: int | None = None
                tail_distance: int | None = None
                for hi in (0, 1):
                    stack: list[CellLane] = stacks[hi]
                    for idx, segment in enumerate(stack):
                        seg_r, seg_c, _lane = segment
                        if seg_r == row and seg_c == col:
                            distance_from_tail: int = len(stack) - idx - 1
                            if tail_distance is None or distance_from_tail < tail_distance:
                                chosen_head = hi
                                tail_distance = distance_from_tail
                            break
                if chosen_head is not None:
                    target_color = ci
                    target_head = chosen_head
                    break

        # Occupancy derived from grid or lanes
        if target_color is None:
            if self.env._bridges[row, col]:
                v_code: int = int(self.env._lane_v[row, col])
                h_code: int = int(self.env._lane_h[row, col])
                preferred_code: int = self.sel_color + 1
                lane_code: int = 0
                if preferred_code in {v_code, h_code}:
                    lane_code = preferred_code
                elif v_code > 0:
                    lane_code = v_code
                elif h_code > 0:
                    lane_code = h_code
                if lane_code > 0:
                    target_color = lane_code - 1
            else:
                color_code: int = int(self.env._grid_codes[row, col])
                if color_code > 0:
                    target_color = color_code - 1

            if target_color is not None:
                head0: Coord = self.env._heads[target_color][0]
                head1: Coord = self.env._heads[target_color][1]
                dist0: int = self.env._metric((head0[0], head0[1]), (row, col))
                dist1: int = self.env._metric((head1[0], head1[1]), (row, col))
                target_head = 0 if dist0 <= dist1 else 1

        if target_color is None:
            return False

        self.sel_color = target_color
        if target_head is not None:
            self.sel_head = target_head

        return (self.sel_color != initial_color) or (self.sel_head != initial_head)

    def _handle_mouse_cell(self, row: int, col: int) -> None:
        """Handle mouse interaction in path mode.

        Determine the appropriate action based on current head position and target cell.

        :param row: Target row.
        :type row: int
        :param col: Target column.
        :type col: int
        """
        if (
            row < 0
            or row >= self.env.H
            or col < 0
            or col >= self.env.W
            or self.sel_color < 0
            or self.sel_color >= self.env.num_colors
        ):
            return
        self._select_focus_for_cell(row, col)
        self._ensure_color_ready_for_head()
        head_r, head_c = self.env._heads[self.sel_color][self.sel_head]
        dr: int = row - head_r
        dc: int = col - head_c
        # Only move if adjacent
        if abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0):
            d_index: int | None = self._direction_index(dr, dc)
            if d_index is not None:
                base: int = self.sel_color * self.env._actions_per_color + self.sel_head * self.env._num_dirs
                self.env.step(action=base + d_index)

    def _ensure_color_ready_for_head(self) -> None:
        """Reset the currently selected color when switching starting endpoints mid-path."""
        if self.switch_mode:
            return
        color: int = self.sel_color
        head: int = self.sel_head
        stacks: list[list[CellLane]] = self.env._stacks[color]
        selected_stack: list[CellLane] = stacks[head]
        other_stack: list[CellLane] = stacks[1 - head]
        if len(selected_stack) == 1 and len(other_stack) > 1 and not self.env._closed[color]:
            self.env.clear_color_path(color)

    def _draw(self) -> None:
        """Render the environment state into the :mod:`pygame` surface.

        Request an RGB image from :meth:`numberlink.env.NumberLinkRGBEnv._render_rgb` and draw a grid of colored cells.
        Draw a highlighted border for the cursor when in cell mode or for the active head when in path mode.
        Optionally draw a help overlay with key bindings.
        """
        if self.window is None:
            return

        self.window.fill(self.env._render_cfg.grid_background_color)
        img: NDArray[np.uint8] = self.env._render_rgb()
        pygame: ModuleType = self.pygame

        # Blit the env RGB image scaled to the window. make_surface expects (W, H, 3)
        arr_for_surface: NDArray[np.uint8] = np.transpose(img, (1, 0, 2))
        surface: Surface = pygame.surfarray.make_surface(arr_for_surface)
        scaled_surface: Surface = pygame.transform.scale(surface, (self.env.W * self.cell, self.env.H * self.cell))
        self.window.blit(scaled_surface, (0, 0))

        # Draw gridlines if configured and not already drawn into the RGB image
        if self.env._render_cfg.gridline_color is not None and (
            self.env._pixels_per_cell_h < 2 or self.env._pixels_per_cell_w < 2
        ):
            gridline_color: RGBInt = self.env._render_cfg.gridline_color
            gridline_thickness: int = self.env._render_cfg.gridline_thickness
            # Vertical lines
            for c in range(self.env.W + 1):
                x: int = c * self.cell
                pygame.draw.line(self.window, gridline_color, (x, 0), (x, self.env.H * self.cell), gridline_thickness)
            # Horizontal lines
            for r in range(self.env.H + 1):
                y: int = r * self.cell
                pygame.draw.line(self.window, gridline_color, (0, y), (self.env.W * self.cell, y), gridline_thickness)

        # Draw endpoint numbers with pygame only when env image doesn't have enough pixels-per-cell
        if self.env._render_cfg.show_endpoint_numbers and (
            self.env._pixels_per_cell_h < 10 or self.env._pixels_per_cell_w < 10
        ):
            specs: list[tuple[str, Coord, RGBInt, int]] = build_endpoint_labels(
                endpoints=self.env._endpoints,
                pixels_per_cell_h=self.cell,
                pixels_per_cell_w=self.cell,
                min_scale=self.env._render_cfg.number_font_min_scale,
                max_scale=self.env._render_cfg.number_font_max_scale,
                gridline_thickness=self.env._render_cfg.gridline_thickness or 0,
            )
            for text, center, _color, scale in specs:
                text_s: str = text
                center_px: Coord = (center[0], center[1])
                color_rgb: RGBInt = (
                    self.env._render_cfg.number_font_color[0],
                    self.env._render_cfg.number_font_color[1],
                    self.env._render_cfg.number_font_color[2],
                )
                border_rgb: RGBInt = (
                    self.env._render_cfg.number_font_border_color[0],
                    self.env._render_cfg.number_font_border_color[1],
                    self.env._render_cfg.number_font_border_color[2],
                )
                scale_i: int = scale
                self._draw_text_centered(
                    text=text_s,
                    center=center_px,
                    color=color_rgb,
                    scale=scale_i,
                    outline_color=border_rgb,
                    outline_thickness=self.env._render_cfg.number_font_border_thickness,
                )

        # Draw cursor or head highlight
        if self.switch_mode:
            row, col = self.cursor
            if 0 <= row < self.env.H and 0 <= col < self.env.W:
                rect: Rect = pygame.Rect(col * self.cell, row * self.cell, self.cell, self.cell)
                border_color: RGBInt
                if self.env._endpoint_mask[row, col]:
                    border_color = self.env._render_cfg.cursor_endpoint_highlight_color
                elif 0 <= self.sel_color < self.env.num_colors:
                    palette: NDArray[np.uint8] = self.env._palette_stack[self.sel_color]
                    border_color = (int(palette[0]), int(palette[1]), int(palette[2]))
                else:
                    border_color = (255, 255, 255)  # Fallback white color
                pygame.draw.rect(
                    self.window, border_color, rect, width=max(1, self.env._render_cfg.cursor_highlight_thickness)
                )
        elif 0 <= self.sel_color < self.env.num_colors:
            hr, hc = self.env._heads[self.sel_color][self.sel_head]
            if 0 <= hr < self.env.H and 0 <= hc < self.env.W:
                rect = pygame.Rect(hc * self.cell, hr * self.cell, self.cell, self.cell)
                pygame.draw.rect(
                    self.window,
                    self.env._render_cfg.active_head_highlight_color,
                    rect,
                    width=max(1, self.env._render_cfg.active_head_highlight_thickness),
                )

        # Draw help overlay if enabled
        if self.show_help:
            self._draw_help_overlay()

        solved: bool = self.env._is_solved()
        action_mask: NDArray[np.uint8] = self.env._compute_action_mask()
        deadlocked: bool = self.env._is_deadlocked(action_mask, solved)
        truncated: bool = self.env._steps >= self.env.max_steps and not solved and not deadlocked
        if solved or deadlocked or truncated:
            self._draw_status_overlay(solved, deadlocked, truncated)

        # Draw replay status/info overlays
        self._draw_replay_overlay()

    @staticmethod
    def _metrics_for_scale(scale: int, *, line_spacing: int) -> RGBInt:
        """Return character cell width, char height, and line spacing for a given scale.

        This reproduces the simple metrics logic previously defined as nested helpers in the viewer methods.
        The return order matches the previous (cw, ch, ls).
        """
        cw: int = (5 + 1) * scale
        ch: int = 7 * scale
        ls: int = line_spacing
        return cw, ch, ls

    @staticmethod
    def _wrap_lines(lines: list[str], cw: int, max_w: int, *, padding: int) -> list[str]:
        """Wrap lines to fit within max_w using character cell width `cw` and panel padding.

        This implements the same wrapping algorithm used in the nested helpers. The `padding` parameter is required to
        ensure identical max_chars computation.
        """
        wrapped: list[str] = []
        max_chars: int = max(1, (max_w - 2 * padding - 10) // cw)
        for line in lines:
            if len(line) <= max_chars:
                wrapped.append(line)
            else:
                words: list[str] = line.split(" ")
                cur: str = ""
                for w in words:
                    add: str = w if not cur else " " + w
                    if len(cur) + len(add) <= max_chars:
                        cur += add
                    else:
                        if cur:
                            wrapped.append(cur)
                        cur = w
                if cur:
                    wrapped.append(cur)
        return wrapped

    def _draw_help_overlay(self) -> None:
        """Draw a help overlay showing key bindings on the game window."""
        if self.window is None:
            return

        help_lines: list[str] = [
            "Press H to toggle help",
            "Press R to reset",
            "Press ESC to exit",
            "",
            "TAB / Shift+TAB: cycle colors",
        ]

        help_lines.extend(
            [
                "Arrow keys: move cursor",
                "SPACE/ENTER: paint cell",
                "0: clear cell",
                "1-9: select color",
                "Mouse: click and drag",
            ]
            if self.switch_mode
            else [
                "Arrow keys: move head",
                "[ / ]: select left/right head",
                "SPACE: backtrack one step",
                "Mouse: click and drag",
            ]
        )

        if self.env.variant.allow_diagonal:
            help_lines.append("Q/E/Z/C: diagonal moves")

        # Solution replay controls
        help_lines.extend(["T: play/pause solution", "S: stop replay"])
        # Background panel sized to content using bitmap font metrics and wrapping to fit window
        padding: int = 10
        text_scale: int = 3
        max_panel_w: int = self.env.W * self.cell - 20
        max_panel_h: int = self.env.H * self.cell - 20

        # Try scales 3 down to 2 to fit in window
        for candidate_scale in (3, 2):
            cw, ch, ls = self._metrics_for_scale(candidate_scale, line_spacing=4)
            wrapped: list[str] = self._wrap_lines(help_lines, cw, max_panel_w, padding=padding)
            line_h: int = ch + ls
            panel_w: int = min(max_panel_w, max((len(s) * cw for s in wrapped), default=0) + 2 * padding + 10)
            panel_h: int = min(max_panel_h, len(wrapped) * line_h + 2 * padding)
            if panel_w <= max_panel_w and panel_h <= max_panel_h:
                text_scale = candidate_scale
                help_lines = wrapped
                break

        # Compose overlay and draw
        cw, ch, ls = self._metrics_for_scale(text_scale, line_spacing=4)
        line_h = ch + ls
        panel_w = min(max_panel_w, max((len(s) * cw for s in help_lines), default=0) + 2 * padding + 10)
        panel_h = min(max_panel_h, len(help_lines) * line_h + 2 * padding)

        overlay: Surface = self.pygame.Surface((panel_w, panel_h))
        overlay.set_alpha(self.env._render_cfg.help_overlay_background_alpha)
        overlay.fill(self.env._render_cfg.help_overlay_background_color)
        self.window.blit(overlay, (10, 10))

        # Draw text using builtin bitmap font
        y_offset: int = 10 + padding
        for line in help_lines:
            self._draw_text(
                text=line,
                topleft=(20, y_offset),
                color=self.env._render_cfg.help_overlay_font_color,
                scale=text_scale,
                outline_color=self.env._render_cfg.help_overlay_font_border_color,
                outline_thickness=self.env._render_cfg.help_overlay_font_border_thickness,
            )
            y_offset += line_h

    def _draw_status_overlay(self, solved: bool, deadlocked: bool, truncated: bool) -> None:
        """Draw an end-of-game overlay summarizing the current status and available actions."""
        if self.window is None:
            return

        if solved:
            header: str = "Solved"
        elif deadlocked:
            header = "Deadlocked"
        elif truncated:
            header = "Step limit reached"
        else:
            header = "Status"

        lines: list[str] = [header, "", "Press R to reset level"]
        if self._can_generate_new_level():
            lines.append("Press N for new level")
        # Hint for solution replay
        if self.env.get_solution():
            lines.append("Press T to play solution")
        padding: int = 16
        max_panel_w: int = self.env.W * self.cell - 20
        max_panel_h: int = self.env.H * self.cell - 20

        text_scale: int = 4
        chosen_lines: list[str] = lines
        origin_x: int = 0
        origin_y: int = 0
        # overlay: Surface | None = None
        pygame: ModuleType = self.pygame
        for candidate_scale in (4, 3, 2):
            cw, ch, ls = self._metrics_for_scale(candidate_scale, line_spacing=6)
            wrapped: list[str] = self._wrap_lines(lines, cw, max_panel_w, padding=padding)
            line_h: int = ch + ls
            panel_w: int = min(max_panel_w, max((len(s) * cw for s in wrapped), default=0) + 2 * padding + 10)
            panel_h: int = min(max_panel_h, len(wrapped) * line_h + 2 * padding)
            # accept fit
            if panel_w <= max_panel_w and panel_h <= max_panel_h:
                text_scale = candidate_scale
                chosen_lines = wrapped
                overlay: Surface = pygame.Surface((panel_w, panel_h))
                overlay.set_alpha(self.env._render_cfg.help_overlay_background_alpha)
                overlay.fill(self.env._render_cfg.help_overlay_background_color)
                origin_x = max(0, (self.env.W * self.cell - panel_w) // 2)
                origin_y = max(0, (self.env.H * self.cell - panel_h) // 2)
                self.window.blit(overlay, (origin_x, origin_y))
                break

        # Draw text
        cw, ch, ls = self._metrics_for_scale(text_scale, line_spacing=6)
        line_h = ch + ls
        y_offset: int = origin_y + padding
        text_x: int = origin_x + padding + 10
        for line in chosen_lines:
            self._draw_text(
                text=line,
                topleft=(text_x, y_offset),
                color=self.env._render_cfg.help_overlay_font_color,
                scale=text_scale,
                outline_color=self.env._render_cfg.help_overlay_font_border_color,
                outline_thickness=self.env._render_cfg.help_overlay_font_border_thickness,
            )
            y_offset += line_h

    def _draw_text(
        self,
        text: str,
        topleft: Coord,
        color: RGBInt,
        scale: int = 3,
        outline_color: RGBInt | None = (0, 0, 0),
        outline_thickness: int = 1,
    ) -> int:
        """Draw text at topleft (x,y) onto self.window using the bitmap font.

        Returns the width in pixels of the rendered string.
        """
        # Guard so static type checkers know pygame and window are present.
        if self.window is None:
            return 0

        x0, y0 = topleft
        x: int = x0
        y: int = y0
        # If outline thickness is <= 0 or outline color is None, suppress outline rendering
        outline: RGBInt | None = None if (outline_thickness <= 0 or outline_color is None) else outline_color
        for ch in text:
            ch_u: str = ch.upper()

            # Compute absolute pixel origin for this glyph
            glyph_x0: int = x
            glyph_y0: int = y

            # Blit cached glyph surface (faster than per-pixel draws)
            fg: RGBInt = (color[0], color[1], color[2])
            outline_rgb: RGBInt | None = None if outline is None else (outline[0], outline[1], outline[2])
            surf: Surface = self._get_glyph_surface(ch_u, scale, max(0, outline_thickness), fg, outline_rgb)
            # blit surf at glyph_x0, glyph_y0
            self.window.blit(source=surf, dest=(glyph_x0, glyph_y0))

            x += (5 + 1) * scale
        return x - x0

    def _draw_text_centered(
        self,
        text: str,
        center: Coord,
        color: RGBInt,
        scale: int = 3,
        outline_color: RGBInt | None = (0, 0, 0),
        outline_thickness: int = 1,
    ) -> None:
        # Approximate text width/height
        width: int = len(text) * (5 + 1) * scale
        height: int = 7 * scale
        topleft: Coord = (center[0] - width // 2, center[1] - height // 2)
        self._draw_text(text, topleft, color, scale, outline_color, outline_thickness)

    def _get_glyph_surface(
        self, ch: str, scale: int, outline_thickness: int, fg: RGBInt, outline: RGBInt | None
    ) -> Surface:
        """Return a cached pygame Surface for a glyph rendered at the given parameters.

        The surface is tightly sized to the glyph's bounding box.
        """
        key: tuple[str, int, int, RGBInt, RGBInt | None] = (
            ch,
            scale,
            outline_thickness,
            (fg[0], fg[1], fg[2]),
            None if outline is None else (outline[0], outline[1], outline[2]),
        )
        if key in self._glyph_cache:
            # mark as recently used
            self._glyph_cache.move_to_end(key)
            return self._glyph_cache[key]

        # Build glyph mask (reuse bitmap logic)
        rows: tuple[int, ...] = BITMAP_FONT.get(ch.upper(), BITMAP_FONT[" "])
        glyph_h: int = 7 * scale
        glyph_w: int = 5 * scale
        mask: NDArray[np.bool_] = np.zeros((glyph_h, glyph_w), dtype=np.bool_)
        for ry, rowbits in enumerate(rows):
            if rowbits == 0:
                continue
            for cx in range(5):
                if (rowbits >> (4 - cx)) & 1:
                    y0s: int = ry * scale
                    x0s: int = cx * scale
                    mask[y0s : y0s + scale, x0s : x0s + scale] = True

        # create surface with per-pixel alpha
        pygame: ModuleType = self.pygame
        if pygame.get_init() is False:
            raise RuntimeError("pygame not initialized for viewer glyph surface creation")

        surf: Surface = pygame.Surface((glyph_w, glyph_h), pygame.SRCALPHA)
        surf.fill(color=(0, 0, 0, 0))

        # Prepare outline only if requested
        union_dil: NDArray[np.bool_]
        inner_ring: NDArray[np.bool_]
        if outline is not None and outline_thickness > 0:
            # Outer dilation union for external border
            current: NDArray[np.bool_] = mask.copy()
            union_dil = np.zeros_like(mask)
            for _ in range(outline_thickness):
                pad: NDArray[np.bool_] = np.pad(current, 1, mode="constant", constant_values=False)
                dil = (
                    pad[1:-1, 1:-1]
                    | pad[:-2, 1:-1]
                    | pad[2:, 1:-1]
                    | pad[1:-1, :-2]
                    | pad[1:-1, 2:]
                    | pad[:-2, :-2]
                    | pad[:-2, 2:]
                    | pad[2:, :-2]
                    | pad[2:, 2:]
                )
                union_dil |= dil
                current = dil

            # Inner erosion ring for internal borders near holes
            current_e: NDArray[np.bool_] = mask.copy()
            for _ in range(outline_thickness):
                pad_e: NDArray[np.bool_] = np.pad(current_e, 1, mode="constant", constant_values=False)
                ero = (
                    pad_e[1:-1, 1:-1]
                    & pad_e[:-2, 1:-1]
                    & pad_e[2:, 1:-1]
                    & pad_e[1:-1, :-2]
                    & pad_e[1:-1, 2:]
                    & pad_e[:-2, :-2]
                    & pad_e[:-2, 2:]
                    & pad_e[2:, :-2]
                    & pad_e[2:, 2:]
                )
                current_e = ero
            inner_ring = mask & (~current_e)
        else:
            union_dil = np.zeros_like(mask)
            inner_ring = np.zeros_like(mask)

        # outside-connected mask and border derivation
        border_outer: NDArray[np.bool_] = np.zeros_like(mask)
        if outline is not None and outline_thickness > 0:
            # outside-connected mask ensures outline doesn't flood interior holes
            outside: NDArray[np.bool_] = ~mask
            outside_conn: NDArray[np.bool_] = np.zeros_like(outside)
            outside_conn[0, :] = outside[0, :]
            outside_conn[-1, :] = outside[-1, :]
            outside_conn[:, 0] = outside[:, 0]
            outside_conn[:, -1] = outside[:, -1]
            while True:
                pad_oc: NDArray[np.bool_] = np.pad(outside_conn, 1, mode="constant", constant_values=False)
                nbr = (
                    pad_oc[1:-1, 1:-1]
                    | pad_oc[:-2, 1:-1]
                    | pad_oc[2:, 1:-1]
                    | pad_oc[1:-1, :-2]
                    | pad_oc[1:-1, 2:]
                    | pad_oc[:-2, :-2]
                    | pad_oc[:-2, 2:]
                    | pad_oc[2:, :-2]
                    | pad_oc[2:, 2:]
                )
                new_out = outside_conn | (nbr & outside)
                if new_out.sum() == outside_conn.sum():
                    break
                outside_conn = new_out

            # External outline: dilated union minus mask, constrained to outside-connected
            border_outer = union_dil & outside_conn & (~mask)

        # Combine external and internal borders
        border_mask: NDArray[np.bool_] = border_outer | inner_ring

        # draw border then fill onto surf
        if outline is not None and border_mask.any():
            ys: NDArray[np.intp]
            xs: NDArray[np.intp]
            ys, xs = np.nonzero(border_mask)
            for yy, xx in zip(ys, xs, strict=True):
                surf.set_at((xx, yy), (*outline, 255))

        # Fill foreground excluding inner outline ring to keep inner outline visible
        fill_mask: NDArray[np.bool_] = mask & (~inner_ring)
        if fill_mask.any():
            ys, xs = np.nonzero(fill_mask)
            for yy, xx in zip(ys, xs, strict=True):
                surf.set_at((xx, yy), (*fg, 255))

        # insert and enforce LRU cap
        self._glyph_cache[key] = surf
        # keep cache bounded
        if len(self._glyph_cache) > GLYPH_CACHE_MAX:
            # pop the oldest item
            self._glyph_cache.popitem(last=False)

        return surf

    def _draw_replay_overlay(self) -> None:
        """Draw a small overlay indicating replay status or transient info messages."""
        if self.window is None:
            return

        # Transient info (e.g., no-solution)
        if self._replay_info_message is not None:
            msg: str = self._replay_info_message
            padding: int = 10
            text_scale: int = 3
            max_w: int = self.env.W * self.cell - 20
            max_h: int = self.env.H * self.cell - 20
            char_w: int = (5 + 1) * text_scale
            char_h: int = 7 * text_scale
            # Wrap if needed
            max_chars: int = max(1, (max_w - 2 * padding - 10) // char_w)
            lines: list[str] = []
            if len(msg) <= max_chars:
                lines = [msg]
            else:
                cur: str = ""
                for w in msg.split(" "):
                    add: str = w if not cur else " " + w
                    if len(cur) + len(add) <= max_chars:
                        cur += add
                    else:
                        if cur:
                            lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)
            line_h: int = char_h + 4
            panel_w: int = min(max_w, max((len(s) * char_w for s in lines), default=0) + 2 * padding + 10)
            panel_h: int = min(max_h, len(lines) * line_h + 2 * padding)
            overlay: Surface = self.pygame.Surface((panel_w, panel_h))
            overlay.set_alpha(self.env._render_cfg.help_overlay_background_alpha)
            overlay.fill(self.env._render_cfg.help_overlay_background_color)
            # bottom-left corner
            origin: Coord = (10, self.env.H * self.cell - panel_h - 10)
            self.window.blit(overlay, origin)
            y: int = origin[1] + padding
            for line in lines:
                self._draw_text(
                    text=line,
                    topleft=(origin[0] + padding + 10, y),
                    color=self.env._render_cfg.help_overlay_font_color,
                    scale=text_scale,
                    outline_color=self.env._render_cfg.help_overlay_font_border_color,
                    outline_thickness=self.env._render_cfg.help_overlay_font_border_thickness,
                )
                y += line_h

            return

        # Replay status (playing/paused)
        if self._replay_state in {"playing", "paused"}:
            status: str = "Playing" if self._replay_state == "playing" else "Paused"
            total: int = 0 if self._replay_solution is None else len(self._replay_solution)
            idx: int = min(self._replay_index, total)
            msg = f"Solution replay: {status} ({idx}/{total}) - T: play/pause, S: stop"
            padding = 10
            text_scale = 3
            max_w = self.env.W * self.cell - 20
            char_w = (5 + 1) * text_scale
            char_h = 7 * text_scale
            # Wrap
            max_chars = max(1, (max_w - 2 * padding - 10) // char_w)
            lines = []
            if len(msg) <= max_chars:
                lines = [msg]
            else:
                cur = ""
                for w in msg.split(" "):
                    add = w if not cur else " " + w
                    if len(cur) + len(add) <= max_chars:
                        cur += add
                    else:
                        if cur:
                            lines.append(cur)
                        cur = w
                if cur:
                    lines.append(cur)
            line_h = char_h + 4
            panel_w = min(max_w, max((len(s) * char_w for s in lines), default=0) + 2 * padding + 10)
            panel_h = len(lines) * line_h + 2 * padding
            overlay = self.pygame.Surface((panel_w, panel_h))
            overlay.set_alpha(self.env._render_cfg.help_overlay_background_alpha)
            overlay.fill(self.env._render_cfg.help_overlay_background_color)
            origin = (10, self.env.H * self.cell - panel_h - 10)
            self.window.blit(overlay, origin)
            y = origin[1] + padding
            for line in lines:
                self._draw_text(
                    text=line,
                    topleft=(origin[0] + padding + 10, y),
                    color=self.env._render_cfg.help_overlay_font_color,
                    scale=text_scale,
                    outline_color=self.env._render_cfg.help_overlay_font_border_color,
                    outline_thickness=self.env._render_cfg.help_overlay_font_border_thickness,
                )
                y += line_h


def _detect_notebook_environment() -> str:
    """Return the active notebook environment identifier or ``"none"`` when not in a notebook."""
    try:
        from IPython.core.getipython import get_ipython  # noqa: PLC0415
    except ImportError:
        return "none"

    shell: InteractiveShell | None = get_ipython()
    if shell is None:
        return "none"

    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"

    shell_name: str = shell.__class__.__name__
    shell_module: str = shell.__class__.__module__
    if shell_name == "ZMQInteractiveShell":
        return "jupyter"

    if shell_module.startswith("ipykernel.") or getattr(shell, "kernel", None) is not None:
        return "jupyter"

    return "none"


def _show_notebook_missing_message(env_label: str, detail: str | None = None) -> None:
    """Surface a user friendly notice when notebook dependencies are missing."""
    base_message: str = (
        "Install the optional notebook dependencies with `pip install numberlink[notebook]` to enable inline controls."
    )
    message: str = f"{detail} {base_message}".strip() if detail else base_message

    try:
        from IPython.display import Markdown, display  # noqa: PLC0415
    except ImportError:
        warnings.warn(message, stacklevel=2)
        return

    prefix: str = "Google Colab" if env_label == "colab" else "Notebook"
    display(Markdown(f"> **NumberLink {prefix} support** - {message}"))


def _try_launch_notebook_viewer(base_viewer: NumberLinkViewer) -> bool:
    """Launch the notebook viewer when running in a notebook environment."""
    env_label: str = _detect_notebook_environment()
    if env_label == "none":
        return False

    try:
        from .notebook_viewer import NumberLinkNotebookViewer  # noqa: PLC0415
    except ImportError:
        _show_notebook_missing_message(env_label)
        return True

    try:
        notebook_viewer = NumberLinkNotebookViewer(base_viewer.env, cell_size=base_viewer.cell)
    except RuntimeError as exc:
        _show_notebook_missing_message(env_label, str(exc))
        return True

    notebook_viewer.loop()
    return True
