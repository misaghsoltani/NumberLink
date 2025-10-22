"""Interactive notebook viewer for NumberLink environments.

This module provides a widget based viewer to play and inspect NumberLink puzzles inside Jupyter and Google Colab. It
wraps a :class:`numberlink.env.NumberLinkRGBEnv` instance and exposes keyboard and pointer controls similar to the
pygame viewer included elsewhere in the project. Optional notebook dependencies load lazily to keep the base
installation light. Install extras with ``pip install numberlink[notebook]`` when using the notebook viewer.

The main entry point is :class:`numberlink.notebook_viewer.NumberLinkNotebookViewer`. To render the viewer in a notebook
construct it with a prepared environment and call :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer.loop`.

**Quick start**::

    from numberlink.env import NumberLinkRGBEnv
    from numberlink.notebook_viewer import NumberLinkNotebookViewer

    env = NumberLinkRGBEnv((5, 5))
    viewer = NumberLinkNotebookViewer(env)
    viewer.loop()

.. seealso::
   :class:`numberlink.env.NumberLinkRGBEnv`
      Primary environment that renders RGB frames and exposes the action space used by the viewer.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from .env import NumberLinkRGBEnv

if TYPE_CHECKING:
    from asyncio import Task
    from collections.abc import Callable
    from types import ModuleType

    import gymnasium as gym
    from IPython.display import DisplayHandle
    from ipywidgets import (
        HTML,
        Button,
        Dropdown,
        GridBox,
        HBox,
        Image as IpyImage,
        IntSlider,
        Layout,
        Play,
        ToggleButton,
        ToggleButtons,
        VBox,
    )
    from numpy.typing import NDArray

    from .types import ActType, CellLane, Coord, RGBInt, Snapshot


class DOMEvent(TypedDict, total=True):
    """DOM event payload captured from the notebook front end.

    Events are emitted by ipyevents and forwarded to the viewer. Only fields required by the viewer are
    represented here. Values follow the browser event model and use the relative coordinate system set on the image
    widget.

    :ivar type: DOM event type name such as ``"mousedown"`` or ``"keydown"``.
    :vartype type: str
    :ivar buttons: Mouse button bit mask as reported by the browser when the event fired.
    :vartype buttons: int
    :ivar relativeX: X coordinate relative to the image widget in CSS pixels.
    :vartype relativeX: float
    :ivar relativeY: Y coordinate relative to the image widget in CSS pixels.
    :vartype relativeY: float
    :ivar key: Key value from keyboard events in lower case when available or an empty string.
    :vartype key: str
    :ivar code: Physical key code from keyboard events such as ``"ArrowUp"`` or ``"Numpad5"``.
    :vartype code: str
    :ivar shiftKey: Whether the Shift modifier was active.
    :vartype shiftKey: bool
    :ivar ctrlKey: Whether the Control modifier was active.
    :vartype ctrlKey: bool
    :ivar altKey: Whether the Alt modifier was active.
    :vartype altKey: bool
    """

    type: str
    buttons: int
    relativeX: float
    relativeY: float
    key: str
    code: str
    shiftKey: bool
    ctrlKey: bool
    altKey: bool


class ObserveChange(TypedDict, total=True):
    """Change notification payload for :mod:`ipywidgets` observers.

    This structure mirrors values passed to callback functions registered via ``observe`` on widgets. The viewer relies
    on the ``new`` field to determine the updated value for a setting.

    :ivar new: New value of the observed trait. May be an ``int``, ``str`` or ``None`` depending on the widget.
    :vartype new: int | str | None
    """

    new: int | str | None


class NumberLinkNotebookViewer:
    """Notebook UI for :class:`numberlink.env.NumberLinkRGBEnv`.

    Present an image widget with pointer and keyboard interaction so the puzzle can be played and inspected inside a
    notebook. The viewer supports both path mode and cell switching mode depending on the environment variant
    configuration exposed by ``NumberLinkRGBEnv.variant``.

    The viewer renders frames produced by :meth:`numberlink.env.NumberLinkRGBEnv._render_rgb` and uses
    :meth:`numberlink.env.NumberLinkRGBEnv.step` to apply actions. For cell switching mode it encodes write or clear
    actions with :meth:`numberlink.env.NumberLinkRGBEnv.encode_cell_switching_action`.

    .. note::
       This class requires optional notebook dependencies. Importing or constructing it without the
       ``numberlink[notebook]`` extras raises a :class:`RuntimeError` with details about the missing module.
    """

    def __init__(self, env: NumberLinkRGBEnv | gym.Env[NDArray[np.uint8], np.int64], cell_size: int = 48) -> None:
        """Initialize the viewer and validate inputs.

        Import optional dependencies on demand and normalize the provided environment so that
        :class:`numberlink.env.NumberLinkRGBEnv` is available via ``env``. For wrapped Gym environments the
        underlying unwrapped object must be a :class:`numberlink.env.NumberLinkRGBEnv` instance or a
        :class:`TypeError` is raised.

        :param env: Environment to render. Accepts a direct
            :class:`numberlink.env.NumberLinkRGBEnv` or a Gym environment that wraps one and exposes it via
            ``unwrapped``.
        :param cell_size: Nominal pixel size per grid cell used to compute image layout. Must be positive. Defaults to
            ``48``.

        :raises ValueError: If ``cell_size`` is not positive.
        :raises RuntimeError: If a required optional dependency such as ipyevents or ipywidgets cannot be
            imported. The error message names the missing module and suggests installing ``numberlink[notebook]``.
        :raises TypeError: If ``env`` is neither a :class:`numberlink.env.NumberLinkRGBEnv` nor a Gym environment that
            unwraps to one.
        """
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")

        try:
            import ipyevents  # noqa: PLC0415
            from IPython.display import display  # noqa: PLC0415
            import ipywidgets as widgets  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            missing: str = exc.name or "notebook"
            raise RuntimeError(
                "Notebook viewer requires optional dependencies. Install with `pip install numberlink[notebook]`."
                f" Missing module: {missing}."
            ) from exc

        self._widgets: ModuleType = widgets
        self._display: Callable[..., DisplayHandle | None] = display
        self._ipyevents: ModuleType = ipyevents
        self._Image: ModuleType = Image
        self.env: NumberLinkRGBEnv
        if isinstance(env, NumberLinkRGBEnv):
            self.env = env
        else:
            unwrapped: NumberLinkRGBEnv | None = getattr(env, "unwrapped", None)
            if isinstance(unwrapped, NumberLinkRGBEnv):
                self.env = unwrapped
            else:
                raise TypeError("env must be NumberLinkRGBEnv or a gym.Env wrapping one")
        self.cell: int = cell_size

        self.sel_color: int = 0
        self.sel_head: int = 0
        self.switch_mode: bool = self.env.variant.cell_switching_mode
        self.cursor: list[int] = [0, 0]
        self.mouse_dragging: bool = False
        self.last_mouse_cell: Coord | None = None
        self._drag_active: bool = False
        self._suspend_widget_callbacks: bool = False
        self._pixels_per_cell_h: int = max(1, self.env._pixels_per_cell_h)
        self._pixels_per_cell_w: int = max(1, self.env._pixels_per_cell_w)

        self._status_label: HTML = widgets.HTML()
        self._message_label: HTML = widgets.HTML()
        self._image_widget: IpyImage = widgets.Image(format="png")
        self._image_widget.layout.border = "1px solid #b1b5ba"
        self._color_selector: Dropdown = widgets.Dropdown(
            description="Color", layout=widgets.Layout(width="200px"), style={"description_width": "60px"}
        )
        self._prev_color_button: Button = widgets.Button(description="◀", layout=widgets.Layout(width="40px"))
        self._next_color_button: Button = widgets.Button(description="▶", layout=widgets.Layout(width="40px"))
        self._head_selector: ToggleButtons = widgets.ToggleButtons(
            options=[("Head 0", 0), ("Head 1", 1)], layout=widgets.Layout(width="200px"), style={"button_width": "96px"}
        )
        self._backtrack_button: Button = widgets.Button(description="Backtrack")
        self._reset_button: Button = widgets.Button(description="Reset")
        self._new_level_button: Button = widgets.Button(description="New level")
        self._auto_paint_toggle: ToggleButton = widgets.ToggleButton(value=False, description="Paint on move")
        self._paint_button: Button = widgets.Button(description="Paint cell")
        self._clear_button: Button = widgets.Button(description="Clear cell")

        self._direction_grid: GridBox = self._build_direction_pad()

        # Help area
        self._help_toggle: ToggleButton = widgets.ToggleButton(value=False, description="Show help")
        self._help_html: HTML = widgets.HTML()
        self._help_html.layout.display = "none"
        self._help_hide_task: Task[None] | None = None

        # Replay controls
        self._replay_state: str = "idle"  # 'idle' | 'playing' | 'paused'
        self._replay_solution: list[ActType] | None = None
        self._replay_index: int = 0
        self._replay_interval_ms: int = 150
        self._pre_replay_snapshot: Snapshot | None = None
        self._replay_info: HTML = widgets.HTML()
        self._play_ctrl: Play = widgets.Play(
            interval=self._replay_interval_ms, value=0, min=0, max=1, step=1, disabled=True
        )
        self._replay_step_btn: Button = widgets.Button(description="Step ▶")
        self._replay_stop_btn: Button = widgets.Button(description="Stop ⏹")
        self._replay_restore_toggle: ToggleButton = widgets.ToggleButton(value=True, description="Restore on stop")
        self._replay_speed: IntSlider = widgets.IntSlider(
            description="Speed (ms)",
            min=50,
            max=600,
            step=10,
            value=self._replay_interval_ms,
            layout=widgets.Layout(width="280px"),
            style={"description_width": "90px"},
        )

        self._link_callbacks()
        self._attach_pointer_events()
        self._attach_keyboard_events()
        self._reset_view_state()
        self._compose_layout()
        self._refresh_status()
        self._refresh_frame()

    # Public API

    def loop(self) -> None:
        """Display the viewer inside the current notebook cell.

        Call this method in a notebook cell to render the widget tree. It refreshes status and frame and then displays
        ``_root`` using :func:`IPython.display.display`.

        :return: ``None``
        """
        self._refresh_status()
        self._refresh_frame()
        self._display(self._root)

    # UI composition

    def _compose_layout(self) -> None:
        """Compose the widget layout and labels.

        Build the color selection row, head selection row, direction pad and utility controls. This method also prepares
        the help content and replay controls. Call it during initialization before rendering.

        :return: ``None``
        """
        widgets: ModuleType = self._widgets

        color_row: HBox = widgets.HBox([self._prev_color_button, self._color_selector, self._next_color_button])

        head_row: HBox = widgets.HBox([self._head_selector, self._backtrack_button])

        utility_buttons: list[Button] = [self._reset_button]
        if self._can_generate_new_level():
            utility_buttons.append(self._new_level_button)
        utility_row: HBox = widgets.HBox(utility_buttons)

        self._cell_tools_row: HBox = widgets.HBox([self._auto_paint_toggle, self._paint_button, self._clear_button])
        self._cell_tools_row.layout.display = "none" if not self.switch_mode else "flex"

        # Help text content
        self._help_html.value = self._build_help_html()

        # Replay row
        self._replay_row: HBox = widgets.HBox([
            self._play_ctrl,
            self._replay_step_btn,
            self._replay_stop_btn,
            self._replay_speed,
            self._replay_restore_toggle,
        ])
        self._replay_row.layout.display = "flex"

        self._root: VBox = widgets.VBox([
            self._status_label,
            self._image_widget,
            self._message_label,
            self._replay_row,
            self._replay_info,
            color_row,
            head_row,
            self._direction_grid,
            self._cell_tools_row,
            utility_row,
            self._help_toggle,
            self._help_html,
        ])

    def make_button(self, label: str, size: Layout, dr: int, dc: int, *, enabled: bool) -> Button:
        """Create a direction button for the direction pad.

        The returned button invokes :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._handle_direction` with
        the provided row and column deltas.

        :param label: Text label to display on the button.
        :param size: Widget layout for width and height.
        :param dr: Row delta applied when clicked.
        :param dc: Column delta applied when clicked.
        :param enabled: Whether the button is active. Disabled buttons do not fire actions.

        :return: Configured ipywidgets Button instance.
        """
        widgets: ModuleType = self._widgets

        btn: Button = widgets.Button(description=label, layout=size, disabled=not enabled)

        def _on_click(_: Button) -> None:
            """Handle a direction button click.

            Invoke :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._handle_direction` with the fixed deltas
            specified by the outer closure.
            """
            self._handle_direction(dr, dc)

        btn.on_click(_on_click)
        return btn

    def _build_direction_pad(self) -> GridBox:
        """Build the three by three direction pad.

        Creates a grid of directional buttons with optional diagonals depending on
        ``NumberLinkRGBEnv.variant.allow_diagonal``.

        :return: ipywidgets GridBox widget containing the direction pad.
        """
        widgets: ModuleType = self._widgets
        allow_diagonal: bool = self.env.variant.allow_diagonal
        size: Layout = widgets.Layout(width="42px", height="42px")

        pad_children: list[Button] = [
            self.make_button("↖", size, -1, -1, enabled=allow_diagonal),
            self.make_button("↑", size, -1, 0, enabled=True),
            self.make_button("↗", size, -1, 1, enabled=allow_diagonal),
            self.make_button("←", size, 0, -1, enabled=True),
            widgets.Button(description="·", disabled=True, layout=size),
            self.make_button("→", size, 0, 1, enabled=True),
            self.make_button("↙", size, 1, -1, enabled=allow_diagonal),
            self.make_button("↓", size, 1, 0, enabled=True),
            self.make_button("↘", size, 1, 1, enabled=allow_diagonal),
        ]

        grid_layout: Layout = widgets.Layout(grid_template_columns="repeat(3, 46px)", grid_gap="4px")
        return widgets.GridBox(pad_children, layout=grid_layout)

    def _link_callbacks(self) -> None:
        """Connect widget callbacks to viewer handlers.

        Wire widget events like clicks and value changes to methods such as
        :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._handle_reset` and
        :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._on_color_change`. Also install replay controls and
        the help panel toggle behavior.

        :return: ``None``
        """

        def wrap(handler: Callable[[], None]) -> Callable[[Button], None]:
            def _inner(_: Button) -> None:
                """Invoke a zero argument handler from a Button callback.

                ``Button.on_click`` passes the clicked button to the callback. This wrapper ignores the argument and
                calls the provided zero argument handler.
                """
                handler()

            return _inner

        self._prev_color_button.on_click(wrap(lambda: self._change_color(-1)))
        self._next_color_button.on_click(wrap(lambda: self._change_color(1)))
        self._color_selector.observe(self._on_color_change, names="value")
        self._head_selector.observe(self._on_head_change, names="value")
        self._backtrack_button.on_click(wrap(self._handle_backtrack))
        self._reset_button.on_click(wrap(self._handle_reset))
        self._new_level_button.on_click(wrap(self._handle_new_level))
        self._paint_button.on_click(wrap(self._handle_paint))
        self._clear_button.on_click(wrap(self._handle_clear))

        # Help toggle
        def _on_help_toggled(_: ToggleButton) -> None:
            self._help_html.layout.display = "block" if self._help_toggle.value else "none"
            if self._help_toggle.value:
                try:
                    if self._help_hide_task is not None and not self._help_hide_task.done():
                        self._help_hide_task.cancel()
                except Exception:
                    pass
                self._help_hide_task = asyncio.create_task(self._auto_hide_help())

        self._help_toggle.observe(lambda change: _on_help_toggled(self._help_toggle), names="value")

        # Replay controls
        self._play_ctrl.observe(self._on_play_value_change, names="value")
        self._replay_step_btn.on_click(wrap(self._replay_step_once))
        self._replay_stop_btn.on_click(wrap(self._replay_stop))
        self._replay_speed.observe(self._on_replay_speed_change, names="value")

    def _attach_pointer_events(self) -> None:
        """Attach mouse or touch event handlers to the image widget.

        Register an ipyevents Event to capture pointer activity using the ``relative`` coordinate system. The handler
        forwards events to :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._handle_pointer_event`.

        :return: ``None``
        """
        self._pointer_events: ModuleType = self._ipyevents.Event(
            source=self._image_widget,
            watched_events=["mousedown", "mousemove", "mouseup", "mouseleave", "click"],
            prevent_default_action=True,
            coordinate_system="relative",
        )
        self._pointer_events.on_dom_event(self._handle_pointer_event)

    def _attach_keyboard_events(self) -> None:
        """Attach keyboard event handlers to the image widget.

        The image captures keyboard focus when clicked so arrow keys and shortcuts are delivered to the viewer. Events
        are handled by :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._handle_key_event`.

        :return: ``None``
        """
        self._keyboard_events: ModuleType = self._ipyevents.Event(
            source=self._image_widget, watched_events=["keydown"], prevent_default_action=True
        )
        self._keyboard_events.on_dom_event(self._handle_key_event)

    # Event handlers

    def _handle_pointer_event(self, event: DOMEvent) -> None:
        """Handle a pointer event from the image widget.

        Interpret button presses, drags and clicks to update the cursor in cell mode or to attempt a path step in path
        mode. Coordinate extraction is delegated to
        :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._handle_pointer_coordinates`.

        :param event: Event payload received from ipyevents.
        """
        event_type: str | None = event["type"]
        if not event_type:
            return

        buttons_int: int = event["buttons"]

        if event_type == "mousedown" and buttons_int == 1:
            self._drag_active = True
            self._handle_pointer_coordinates(event)
        elif event_type == "mousemove" and self._drag_active:
            self._handle_pointer_coordinates(event)
        elif event_type in {"mouseup", "mouseleave"}:
            self._drag_active = False
        elif event_type == "click":
            self._handle_pointer_coordinates(event)

    def _handle_key_event(self, event: DOMEvent) -> None:
        """Handle a keyboard event when the image widget has focus.

        Support movement keys, digit selection in cell mode, backtracking and utility shortcuts. This method updates
        the environment through :meth:`numberlink.env.NumberLinkRGBEnv.step` and refreshes the UI via
        :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._after_action`.

        :param event: Event payload containing key and modifier information.
        """
        key: str | None = event["key"]
        code: str | None = event["code"]
        if not key and not code:
            return

        k: str = (key or "").lower()
        shift: bool = event["shiftKey"]

        # Common keys
        if k == "tab":
            self._cycle_color(-1 if shift else 1)
            self._after_action()
            return

        if self.switch_mode:
            # Movement
            move_map: dict[str, Coord] = {
                "arrowup": (-1, 0),
                "arrowright": (0, 1),
                "arrowdown": (1, 0),
                "arrowleft": (0, -1),
                "q": (-1, -1),
                "e": (-1, 1),
                "z": (1, -1),
                "c": (1, 1),
            }
            if (key or "").lower() in move_map:
                dr, dc = move_map[(key or "").lower()]
                self._move_cursor(dr, dc)
                if self._auto_paint_toggle.value:
                    self._paint_selected_cell()
                self._after_action()
                return

            # Digit selection 0 clears, 1..9 selects and (optionally) paints
            if k in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"} or (code or "").lower().startswith("numpad"):
                digit = 0
                try:
                    digit = int((key or "0").strip())
                except Exception:
                    try:
                        digit = int((code or "Numpad0").replace("Numpad", ""))
                    except Exception:
                        digit = 0
                if digit == 0:
                    self._clear_selected_cell()
                elif digit - 1 < self.env.num_colors:
                    self.sel_color = digit - 1
                    if self._auto_paint_toggle.value:
                        self._paint_selected_cell()
                self._after_action()
                return

            if k in {" ", "space", "enter", "return"}:
                self._paint_selected_cell()
                self._after_action()
                return
            if k in {"backspace", "delete"}:
                self._clear_selected_cell()
                self._after_action()
                return
        else:
            # Path mode
            move_map = {
                "arrowup": (-1, 0),
                "arrowright": (0, 1),
                "arrowdown": (1, 0),
                "arrowleft": (0, -1),
                "q": (-1, -1),
                "e": (-1, 1),
                "z": (1, -1),
                "c": (1, 1),
            }
            key_lower: str = (key or "").lower()
            if key_lower in move_map:
                dr, dc = move_map[key_lower]
                self._attempt_path_step(dr, dc)
                self._after_action()
                return
            if key in {"[", "]"}:
                # Toggle head
                self.sel_head = 1 - self.sel_head
                self._after_action()
                return
            if k in {" ", "space"}:
                self._backtrack_selected()
                self._after_action()
                return

        # Global utilities
        if k in {"r"}:
            self._handle_reset()
            return
        if k in {"g"} and self._can_generate_new_level():
            self._handle_new_level()
            return
        if k in {"h"}:
            self._help_toggle.value = not self._help_toggle.value
            return

    def _handle_pointer_coordinates(self, event: DOMEvent) -> None:
        """Translate pointer coordinates to grid cell actions.

        Convert relative CSS pixel coordinates to grid indices using the current pixel per cell information. Depending
        on the active mode this either moves the cursor and optionally paints a cell or attempts a path step toward the
        hovered cell.

        :param event: Event payload with relative coordinates.
        """
        x_val: float = event["relativeX"]
        y_val: float = event["relativeY"]
        x: float = float(x_val)
        y: float = float(y_val)

        col = int(float(x) // max(1, self._pixels_per_cell_w))
        row = int(float(y) // max(1, self._pixels_per_cell_h))
        if row < 0 or row >= self.env.H or col < 0 or col >= self.env.W:
            return

        if self.switch_mode:
            self.cursor = [row, col]
            if self._auto_paint_toggle.value or self._drag_active:
                self._paint_selected_cell()
            self._after_action()
        else:
            prior: Coord | None = self.last_mouse_cell
            current: Coord = (row, col)
            if prior != current:
                self.last_mouse_cell = current
                self._handle_mouse_cell(row, col)
                self._after_action()

    def _handle_direction(self, dr: int, dc: int) -> None:
        """Handle a direction input from the pad or keyboard.

        Apply the movement deltas depending on the active mode and trigger a UI refresh through
        :meth:`numberlink.notebook_viewer.NumberLinkNotebookViewer._after_action`.

        :param dr: Row delta to apply.
        :param dc: Column delta to apply.
        """
        if self.switch_mode:
            self._move_cursor(dr, dc)
            if self._auto_paint_toggle.value:
                self._paint_selected_cell()
        else:
            self._attempt_path_step(dr, dc)
        self._after_action()

    async def _auto_hide_help(self) -> None:
        """Auto hide the help panel after a delay.

        Wait for a short duration and then collapse the help area if it is still visible.
        """
        # Auto-hide the help panel after 4 seconds if still visible
        await asyncio.sleep(4.0)
        if self._help_toggle.value:
            self._help_toggle.value = False

    def _change_color(self, delta: int) -> None:
        """Change the selected color by a relative offset and refresh the UI.

        :param delta: Positive or negative offset to apply.
        """
        self._cycle_color(delta)
        self._after_action()

    def _on_color_change(self, change: ObserveChange) -> None:
        """React to color selector changes and update ``sel_color`` when valid.

        :param change: Observe payload from an ipywidgets Dropdown.
        """
        if self._suspend_widget_callbacks:
            return

        # Observe change may provide an int, str or None
        new_value: int | str | None = change.get("new")
        if new_value is None:
            return

        try:
            idx = int(new_value)
        except (TypeError, ValueError):
            return

        if 0 <= idx < self.env.num_colors:
            self.sel_color = idx
            self._after_action()

    def _on_head_change(self, change: ObserveChange) -> None:
        """React to changes in the selected head in path mode.

        :param change: Observe payload from ipywidgets ToggleButtons.
        """
        if self._suspend_widget_callbacks or self.switch_mode:
            return

        # Observe change may provide int, str, or None
        new_val: int | str | None = change.get("new")
        if new_val is None:
            return

        try:
            idx = int(new_val)
        except (TypeError, ValueError):
            return

        if idx in {0, 1}:
            self.sel_head = idx
            self._after_action()

    def _handle_backtrack(self) -> None:
        """Backtrack one step for the selected color and head in path mode."""
        if not self.switch_mode:
            self._backtrack_selected()
            self._after_action()

    def _handle_reset(self) -> None:
        """Reset the environment and viewer state then refresh the UI."""
        self._reset_environment()
        self._after_action()

    def _handle_new_level(self) -> None:
        """Generate a new level when supported by the environment and refresh the UI."""
        if self._can_generate_new_level():
            self._generate_new_level()
            self._after_action()

    def _handle_paint(self) -> None:
        """Paint the currently selected cell in cell switching mode and refresh the UI."""
        if self.switch_mode:
            self._paint_selected_cell()
            self._after_action()

    def _handle_clear(self) -> None:
        """Clear the currently selected cell in cell switching mode and refresh the UI."""
        if self.switch_mode:
            self._clear_selected_cell()
            self._after_action()

    # Viewer state helpers

    def _reset_view_state(self) -> None:
        """Reset viewer local state derived from the environment and clear replay state."""
        self.sel_color = min(0, self.env.num_colors - 1) if self.env.num_colors > 0 else 0
        self.sel_head = 0
        self.switch_mode = self.env.variant.cell_switching_mode
        if self.switch_mode:
            free_cells: NDArray[np.intp] = np.argwhere(~self.env._endpoint_mask)
            if free_cells.size > 0:
                first: NDArray[np.intp] = free_cells[0]
                self.cursor = [int(first[0]), int(first[1])]
            else:
                self.cursor = [0, 0]
        else:
            self.cursor = [0, 0]
        self.mouse_dragging = False
        self.last_mouse_cell = None
        self._drag_active = False

        # Reset replay state and UI
        self._replay_state = "idle"
        self._replay_solution = None
        self._replay_index = 0
        self._pre_replay_snapshot = None
        self._play_ctrl.disabled = True
        self._play_ctrl.value = 0
        self._replay_info.value = ""

    def _can_generate_new_level(self) -> bool:
        """Return whether the environment can generate new levels."""
        return hasattr(self.env, "regenerate_level") and getattr(self.env, "_generator_config", None) is not None

    def _cycle_color(self, delta: int) -> None:
        """Advance the selected color with wrap around and do nothing when there are zero colors.

        :param delta: Offset to add to the current selection. Positive or negative values are allowed.
        """
        if self.env.num_colors <= 0:
            return
        self.sel_color = (self.sel_color + delta) % self.env.num_colors

    def _direction_index(self, dr: int, dc: int) -> int | None:
        """Translate a direction vector to an action index or return ``None`` when not found."""
        for k in range(self.env._num_dirs):
            if self.env._dirs[k][0] == dr and self.env._dirs[k][1] == dc:
                return k
        return None

    def _attempt_path_step(self, dr: int, dc: int) -> None:
        """Attempt a path step in the selected direction if valid."""
        d_index: int | None = self._direction_index(dr, dc)
        if d_index is None or self.sel_color < 0 or self.sel_color >= self.env.num_colors:
            return
        self._ensure_color_ready_for_head()
        base: int = self.sel_color * self.env._actions_per_color + self.sel_head * self.env._num_dirs
        self.env.step(action=base + d_index)

    def _backtrack_selected(self) -> None:
        """Backtrack one step for the active color and head when possible."""
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
        """Move the cursor by ``(dr, dc)`` within the grid bounds."""
        row: int = max(0, min(self.env.H - 1, self.cursor[0] + dr))
        col: int = max(0, min(self.env.W - 1, self.cursor[1] + dc))
        self.cursor = [row, col]

    def _paint_selected_cell(self) -> None:
        """Paint the cell at the cursor with the selected color in cell mode if not already painted."""
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

        desired_code: int = self.sel_color + 1
        if not self.env._bridges[row, col]:
            if int(self.env._grid_codes[row, col]) == desired_code:
                return
        elif int(self.env._lane_v[row, col]) == desired_code and int(self.env._lane_h[row, col]) == desired_code:
            return

        action: int = self.env.encode_cell_switching_action(row, col, desired_code)
        self.env.step(action=action)

    def _clear_selected_cell(self) -> None:
        """Clear any path value from the cell at the cursor in cell mode when not an endpoint."""
        row, col = self.cursor
        if row < 0 or row >= self.env.H or col < 0 or col >= self.env.W or self.env._endpoint_mask[row, col]:
            return

        currently_empty = (
            int(self.env._grid_codes[row, col]) == 0
            and int(self.env._lane_v[row, col]) == 0
            and int(self.env._lane_h[row, col]) == 0
        )
        if currently_empty:
            return

        action: int = self.env.encode_cell_switching_action(row, col, 0)
        self.env.step(action=action)

    def _select_focus_for_cell(self, row: int, col: int) -> bool:
        """Select a color and head that best match the given cell and return whether selection changed."""
        if row < 0 or row >= self.env.H or col < 0 or col >= self.env.W or self.env.num_colors <= 0:
            return False

        initial_color: int = self.sel_color
        initial_head: int = self.sel_head

        target_color: int | None = None
        target_head: int | None = None

        for ci, endpoints in enumerate(self.env._endpoints):
            if endpoints[0] == (row, col):
                target_color = ci
                target_head = 0
                break
            if endpoints[1] == (row, col):
                target_color = ci
                target_head = 1
                break

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
                            distance_from_tail = len(stack) - idx - 1
                            if tail_distance is None or distance_from_tail < tail_distance:
                                chosen_head = hi
                                tail_distance = distance_from_tail
                            break
                if chosen_head is not None:
                    target_color = ci
                    target_head = chosen_head
                    break

        if target_color is None:
            if self.env._bridges[row, col]:
                v_code = int(self.env._lane_v[row, col])
                h_code = int(self.env._lane_h[row, col])
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
                color_code = int(self.env._grid_codes[row, col])
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
        """Handle hover at a specific grid cell in path mode and step when adjacent to the active head."""
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
        if abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0):
            d_index: int | None = self._direction_index(dr, dc)
            if d_index is not None:
                base: int = self.sel_color * self.env._actions_per_color + self.sel_head * self.env._num_dirs
                self.env.step(action=base + d_index)

    def _ensure_color_ready_for_head(self) -> None:
        """Ensure the selected color is prepared for drawing with the active head.

        Clear the other arm when required to keep the path continuous before the next step.
        """
        if self.switch_mode:
            return
        color: int = self.sel_color
        head: int = self.sel_head
        stacks: list[list[CellLane]] = self.env._stacks[color]
        selected_stack: list[CellLane] = stacks[head]
        other_stack: list[CellLane] = stacks[1 - head]
        if len(selected_stack) == 1 and len(other_stack) > 1 and not self.env._closed[color]:
            self.env.clear_color_path(color)

    # Environment helpers

    def _reset_environment(self, seed: int | None = None) -> None:
        """Reset the environment and reinitialize the viewer, optionally seeding the environment."""
        self.env.reset(seed=seed)
        self._reset_view_state()
        self._refresh_status()
        self._refresh_frame()

    def _generate_new_level(self) -> None:
        """Regenerate the current level when supported by the environment and refresh labels and image."""
        try:
            self.env.regenerate_level()
        except Exception:
            return
        self._reset_view_state()
        self._refresh_status()
        self._refresh_frame()

    # Rendering and status updates

    def _after_action(self) -> None:
        """Refresh labels and image after an environment action."""
        self._refresh_status()
        self._refresh_frame()

    def _refresh_status(self) -> None:
        """Update status indicators, widget states and replay controls."""
        mode_label: str = "Cell mode" if self.switch_mode else "Path mode"
        color_label: str = (
            f"Color {self.sel_color + 1}/{self.env.num_colors}" if self.env.num_colors > 0 else "No colors"
        )
        focus_label: str
        focus_label = f"Cursor ({self.cursor[0]}, {self.cursor[1]})" if self.switch_mode else f"Head {self.sel_head}"

        parts: list[str] = [mode_label, color_label, focus_label]
        solved: bool = self.env._is_solved()
        action_mask: NDArray[np.uint8] = self.env._compute_action_mask()
        deadlocked: bool = self.env._is_deadlocked(action_mask, solved)
        truncated: bool = self.env._steps >= self.env.max_steps and not solved and not deadlocked
        if solved:
            parts.append("Solved ✅")
        elif deadlocked:
            parts.append("Deadlock ⛔")
        elif truncated:
            parts.append("Max steps ⏳")

        self._status_label.value = " · ".join(parts)

        self._suspend_widget_callbacks = True
        try:
            if self.env.num_colors > 0:
                options: list[tuple[str, int]] = [(f"Color {i + 1}", i) for i in range(self.env.num_colors)]
                self._color_selector.options = options
                self._color_selector.disabled = False
                self._color_selector.value = min(self.sel_color, self.env.num_colors - 1)
            else:
                self._color_selector.options = [("None", 0)]
                self._color_selector.disabled = True
                self._color_selector.value = 0
        finally:
            self._suspend_widget_callbacks = False

        self._head_selector.disabled = self.switch_mode or self.env.num_colors <= 0
        if not self._head_selector.disabled:
            self._head_selector.value = self.sel_head

        self._cell_tools_row.layout.display = "flex" if self.switch_mode else "none"

        if not self._can_generate_new_level():
            self._new_level_button.disabled = True
        else:
            self._new_level_button.disabled = False

        if solved:
            self._message_label.value = "<span style='color:#1a7f37;'>Puzzle solved!</span>"
        elif deadlocked:
            self._message_label.value = "<span style='color:#bd2c00;'>No valid moves remain (deadlock).</span>"
        elif truncated:
            self._message_label.value = "<span style='color:#bf8700;'>Max step limit reached.</span>"
        else:
            self._message_label.value = (
                "Click the image to focus, then use arrow/diagonal keys, Tab/Shift+Tab, [ ], Space, digits."
            )

        # Enable replay UI when a solution is available
        try:
            sol = self.env.get_solution()
        except Exception:
            sol = None
        has_solution: bool = bool(sol)
        self._play_ctrl.disabled = not has_solution
        self._replay_step_btn.disabled = not has_solution
        self._replay_stop_btn.disabled = not has_solution

    def _refresh_frame(self) -> None:
        """Render the latest RGB frame, highlight selection and push it to the image widget."""
        frame: NDArray[np.uint8] = self.env._render_rgb().copy()
        self._pixels_per_cell_h = max(1, frame.shape[0] // self.env.H)
        self._pixels_per_cell_w = max(1, frame.shape[1] // self.env.W)

        if self.switch_mode:
            row, col = self.cursor
            highlight: RGBInt = self._resolve_color(
                self.env._render_cfg.cursor_endpoint_highlight_color, default=(240, 230, 90)
            )
            thickness: int = max(1, self.env._render_cfg.cursor_highlight_thickness)
            self._draw_cell_border(frame, row, col, highlight, thickness)
        elif 0 <= self.sel_color < self.env.num_colors:
            hr, hc = self.env._heads[self.sel_color][self.sel_head]
            highlight = self._resolve_color(self.env._render_cfg.active_head_highlight_color, default=(255, 255, 255))
            thickness = max(1, self.env._render_cfg.active_head_highlight_thickness)
            self._draw_cell_border(frame, hr, hc, highlight, thickness)

        buffer = io.BytesIO()
        self._Image.fromarray(frame).save(buffer, format="PNG")
        self._image_widget.value = buffer.getvalue()
        self._image_widget.format = "png"
        self._image_widget.width = frame.shape[1]
        self._image_widget.height = frame.shape[0]

        # Update replay info if active
        if self._replay_state != "idle" and self._replay_solution is not None:
            total = len(self._replay_solution)
            self._replay_info.value = f"Replaying step {self._replay_index}/{total}"
        else:
            self._replay_info.value = ""

    def _draw_cell_border(
        self, frame: NDArray[np.uint8], row: int, col: int, color: tuple[int, int, int], thickness: int
    ) -> None:
        """Draw a rectangular border around a grid cell in the rendered frame."""
        if row < 0 or row >= self.env.H or col < 0 or col >= self.env.W:
            return
        cell_h: int = self._pixels_per_cell_h
        cell_w: int = self._pixels_per_cell_w
        r0: int = row * cell_h
        c0: int = col * cell_w
        r1: int = min(frame.shape[0], r0 + cell_h)
        c1: int = min(frame.shape[1], c0 + cell_w)
        if r0 >= r1 or c0 >= c1:
            return

        for t in range(thickness):
            top: int = r0 + t
            bottom: int = r1 - t - 1
            left: int = c0 + t
            right: int = c1 - t - 1
            if top > bottom or left > right:
                break
            frame[top, c0:c1] = color
            frame[bottom, c0:c1] = color
            frame[r0:r1, left] = color
            frame[r0:r1, right] = color

    @staticmethod
    def _resolve_color(color: tuple[int, int, int] | None, *, default: tuple[int, int, int]) -> tuple[int, int, int]:
        """Resolve an optional color to an RGB tuple.

        :param color: Optional tuple in ``(R, G, B)`` order. When ``None`` the provided default is used.
        :param default: Fallback color in ``(R, G, B)`` order used when ``color`` is ``None``.
        :return: Concrete RGB tuple.
        """
        if color is None:
            return default
        return int(color[0]), int(color[1]), int(color[2])

    # ------------------- Replay helpers -------------------

    def _ensure_replay_initialized(self) -> bool:
        """Initialize replay state on demand from the environment solution and return readiness."""
        if self._replay_solution is not None:
            return True
        try:
            solution: list[ActType] | None = self.env.get_solution()
        except Exception:
            solution = None
        if not solution:
            self._replay_info.value = "<span style='color:#bd2c00;'>No solution available for this level.</span>"
            return False
        # Snapshot current env and viewer state
        self._pre_replay_snapshot = self._snapshot_state()
        # Reset to start state
        self.env.reset()
        self._reset_view_state()
        self._replay_solution = solution
        self._replay_index = 0
        # Configure play control range
        self._play_ctrl.min = 0
        self._play_ctrl.max = max(0, len(solution) - 1)
        self._play_ctrl.step = 1
        self._play_ctrl.value = 0
        return True

    def _on_play_value_change(self, change: ObserveChange) -> None:
        """Advance replay according to the play widget value and update the UI."""
        # Each increment advances one action
        if not self._ensure_replay_initialized():
            return
        new_val: int | str | None = change.get("new")
        try:
            target: int = int(new_val) if new_val is not None else self._replay_index
        except Exception:
            target = self._replay_index
        if self._replay_solution is None:
            return
        while self._replay_index <= target and self._replay_index < len(self._replay_solution):
            action = int(self._replay_solution[self._replay_index])
            self.env.step(action=action)
            self._replay_index += 1
        # Auto-stop when finished
        if self._replay_index >= len(self._replay_solution):
            self._replay_state = "paused"
        else:
            self._replay_state = "playing"
        self._after_action()

    def _replay_step_once(self) -> None:
        """Apply a single action from the solution during replay and refresh the frame."""
        if not self._ensure_replay_initialized():
            return
        if self._replay_solution is None:
            return
        if self._replay_index < len(self._replay_solution):
            action = int(self._replay_solution[self._replay_index])
            self.env.step(action=action)
            self._replay_index += 1
            self._after_action()

    def _replay_stop(self) -> None:
        """Stop replay and optionally restore the pre replay snapshot before refreshing the UI."""
        restore = bool(self._replay_restore_toggle.value)
        if restore and self._pre_replay_snapshot is not None:
            self._restore_state(self._pre_replay_snapshot)
        # Reset replay state
        self._replay_state = "idle"
        self._replay_solution = None
        self._replay_index = 0
        self._play_ctrl.value = 0
        self._after_action()

    def _on_replay_speed_change(self, change: ObserveChange) -> None:
        """Update the replay timer interval from the speed slider."""
        try:
            new_val: int | str | None = change.get("new")
            val: int = int(new_val) if new_val is not None else self._replay_interval_ms
        except Exception:
            val = self._replay_interval_ms
        self._replay_interval_ms = val
        self._play_ctrl.interval = max(10, val)

    def _snapshot_state(self) -> Snapshot:
        """Capture a snapshot of environment and viewer state for replay restore and return it."""
        return {
            "_grid_codes": self.env._grid_codes.copy(),
            "_lane_v": self.env._lane_v.copy(),
            "_lane_h": self.env._lane_h.copy(),
            "_heads": [[(r, c) for (r, c) in heads] for heads in self.env._heads],
            "_stacks": [[[(r, c, lane) for (r, c, lane) in arm] for arm in color] for color in self.env._stacks],
            "_closed": self.env._closed.copy(),
            "_steps": self.env._steps,
            "sel_color": self.sel_color,
            "sel_head": self.sel_head,
            "switch_mode": self.switch_mode,
            "cursor": [self.cursor[0], self.cursor[1]],
        }

    def _restore_state(self, snap: Snapshot) -> None:
        """Restore a previously captured snapshot and refresh the UI."""
        self.env._grid_codes[:, :] = snap["_grid_codes"]
        self.env._lane_v[:, :] = snap["_lane_v"]
        self.env._lane_h[:, :] = snap["_lane_h"]
        self.env._heads = [[(r, c) for (r, c) in heads] for heads in snap["_heads"]]
        self.env._stacks = [[[(r, c, lane) for (r, c, lane) in arm] for arm in color] for color in snap["_stacks"]]
        self.env._closed[:] = snap["_closed"]
        self.env._steps = snap["_steps"]
        self.sel_color = snap["sel_color"]
        self.sel_head = snap["sel_head"]
        self.switch_mode = snap["switch_mode"]
        cur: list[int] = snap["cursor"]
        self.cursor = [cur[0], cur[1]]
        self._after_action()

    @staticmethod
    def _build_help_html() -> str:
        """Build the small help panel content as HTML and return it."""
        return (
            "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;'>"
            "<b>Keyboard shortcuts</b><br/>"
            "<u>Common</u>: Tab or Shift+Tab cycle color · R reset · G new level · H toggle help<br/>"
            "<u>Path mode</u>: Arrows or Q E Z C move head · [ or ] switch head · Space backtrack<br/>"
            "<u>Cell mode</u>: Arrows or Q E Z C move cursor · digits 1 2 3 4 5 6 7 8 9 pick color · 0 clears · "
            "Space or Enter paint · Backspace or Delete clear"
            "</div>"
        )


__all__: list[str] = ["NumberLinkNotebookViewer"]
