"""Configuration dataclasses for the Number Link environment.

This module defines frozen dataclasses that configure runtime reward shaping, gameplay
variants, procedural generation parameters, and rendering options. The classes defined
here are consumed by :mod:`numberlink.env` and by :func:`numberlink.generator.generate_level`.

.. seealso::
    :class:`numberlink.config.RewardConfig`
        Reward shaping parameters used by the environment.

    :class:`numberlink.config.VariantConfig`
        Gameplay variant switches used by the generator and environment.

    :class:`numberlink.config.GeneratorConfig`
        Parameters that control procedural level generation.

    :class:`numberlink.config.RenderConfig`
        Options that control RGB rendering produced by the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import RGBInt


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping parameters used by the environment.

    Instances of this dataclass are frozen to allow safe sharing across components.
    Values from this configuration are applied by :mod:`numberlink.env` when
    computing the reward for a timestep.

    :var step_penalty: Per-step penalty applied on every environment step.
    :vartype step_penalty: float
    :var invalid_penalty: Extra penalty applied when an invalid action is taken.
    :vartype invalid_penalty: float
    :var connect_bonus: Bonus awarded when a color connection is completed.
    :vartype connect_bonus: float
    :var win_bonus: Bonus awarded when the puzzle is completed and the win condition is satisfied.
    :vartype win_bonus: float

    .. seealso::
       :class:`numberlink.config.GeneratorConfig`
          Generation parameters that influence how often connections are completed during play.
    """

    step_penalty: float = -0.01
    invalid_penalty: float = -0.05
    connect_bonus: float = 0.5
    win_bonus: float = 5.0


@dataclass(frozen=True)
class VariantConfig:
    """Gameplay variant switches used by the generator and environment.

    Instances of this dataclass control rule variations applied by the level
    generator and by environment logic. For example, enabling ``allow_diagonal``
    changes adjacency checks used by :func:`numberlink.generator._gen_random_walk`
    and distance calculations performed by :func:`numberlink.generator._shortest_dist`.

    :var must_fill: If ``True``, all board cells must be occupied to satisfy the win condition.
    :vartype must_fill: bool
    :var allow_diagonal: If ``True``, diagonal moves are allowed and shortest-distance calculations use Chebyshev
        distance instead of Manhattan distance.
    :vartype allow_diagonal: bool
    :var bridges_enabled: If ``True``, generated puzzles may include bridge cells that permit two lanes of traversal
        through a single grid cell.
    :vartype bridges_enabled: bool
    :var cell_switching_mode: If ``True``, use an alternative action mode in which non-endpoint cells can be assigned
        any color independently of path continuity.
    :vartype cell_switching_mode: bool

    .. seealso::
       :func:`numberlink.generator._shortest_dist`
          Function used to compute distances when ``allow_diagonal`` affects adjacency.
    """

    must_fill: bool = True
    allow_diagonal: bool = False
    bridges_enabled: bool = False
    cell_switching_mode: bool = False


@dataclass(frozen=True)
class GeneratorConfig:
    """Parameters that configure procedural level generation.

    Use this dataclass to select the generation algorithm and to tune constraints
    applied by :func:`numberlink.generator.generate_level`. Supported values for
    ``mode`` are ``'random_walk'`` and ``'hamiltonian'``.

    The ``min_path_length`` field enforces a minimum shortest-path distance between
    endpoints. Distance is computed using Manhattan distance by default and Chebyshev
    distance when the :class:`VariantConfig` has ``allow_diagonal`` set to ``True``.

    :var mode: Generation algorithm identifier. Valid values are ``'random_walk'`` and ``'hamiltonian'``.
    :vartype mode: str
    :var max_retries: Maximum number of attempts the generator will make when retrying placement constraints.
    :vartype max_retries: int
    :var width: Board width in cells.
    :vartype width: int
    :var height: Board height in cells.
    :vartype height: int
    :var colors: Number of color pairs to generate.
    :vartype colors: int
    :var bridges_probability: Probability in the range ``[0.0, 1.0]`` that a bridge cell is added during generation.
    :vartype bridges_probability: float
    :var min_path_length: Minimum shortest-path distance enforced between the endpoints for each generated color pair.
    :vartype min_path_length: int
    :var seed: Optional random seed used for reproducible generation. If ``None``, generation is non-deterministic.
    :vartype seed: int or None

    .. note::
       The procedural Hamiltonian generator does not support bridge placement. See
       :func:`numberlink.generator._gen_hamiltonian_partition` for implementation
       details.
    """

    mode: str = "hamiltonian"
    max_retries: int = 20
    width: int = 8
    height: int = 8
    colors: int = 7
    bridges_probability: float = 0.0
    min_path_length: int = 3
    seed: int | None = None


@dataclass(frozen=True)
class RenderConfig:
    """Rendering options for RGB arrays produced by the viewer.

    Fields in this dataclass control how endpoints and connections are visualized
    when rendering the puzzle as an RGB image. The configuration is consumed by
    :mod:`numberlink.viewer` and by rendering utilities in :mod:`numberlink.render_utils`.

    :var endpoint_border_thickness: Thickness in pixels of the border drawn for endpoints. ``0`` disables borders.
    :vartype endpoint_border_thickness: int
    :var endpoint_border_color: RGB color triple used for endpoint borders, for example ``(255, 255, 255)`` for white.
    :vartype endpoint_border_color: RGBInt
    :var connection_color_adjustment: Integer adjustment to connection colors when endpoint borders are not drawn.
        Positive values brighten colors and negative values darken them.
    :vartype connection_color_adjustment: int
    :var render_height: Total height in pixels of the rendered RGB observation. If ``None``, defaults to one pixel per
        grid row. When specified, each grid cell will be rendered as ``render_height // grid_height`` pixels tall.
    :vartype render_height: int or None
    :var render_width: Total width in pixels of the rendered RGB observation. If ``None``, defaults to one pixel per
        grid column. When specified, each grid cell will be rendered as ``render_width // grid_width`` pixels wide.
    :vartype render_width: int or None
    :var grid_background_color: RGB color triple used for the grid background.
    :vartype grid_background_color: RGBInt
    :var gridline_color: RGB color triple used for grid lines between cells. If ``None``, no gridlines are drawn.
    :vartype gridline_color: RGBInt or None
    :var gridline_thickness: Thickness in pixels of gridlines drawn between cells. Ignored if :attr:`gridline_color`
        is ``None``.
    :vartype gridline_thickness: int
    :var show_endpoint_numbers: If ``True``, endpoint cells display their color index number starting from 0. Endpoints
        of the same color share the same number.
    :vartype show_endpoint_numbers: bool
    :var number_font_color: RGB color triple used for endpoint number text.
    :vartype number_font_color: RGBInt
    :var number_font_border_color: RGB color triple used for number outline.
    :vartype number_font_border_color: RGBInt
    :var number_font_border_thickness: Thickness in pixels for the font outline drawn around endpoint numbers.
    :vartype number_font_border_thickness: int
    :var number_font_min_scale: Minimum integer font scale unit for endpoint labels. If ``None``, automatic sizing
        based on cell pixels is allowed.
    :vartype number_font_min_scale: int
    :var number_font_max_scale: Maximum integer font scale unit for endpoint labels. If ``None``, automatic sizing
        based on cell pixels is allowed.
    :vartype number_font_max_scale: int or None
    :var help_overlay_font_color: RGB color triple used for help overlay text.
    :vartype help_overlay_font_color: RGBInt
    :var help_overlay_font_border_color: RGB color triple used for help overlay text outline.
    :vartype help_overlay_font_border_color: RGBInt
    :var help_overlay_font_border_thickness: Thickness in pixels for the help-overlay font outline.
    :vartype help_overlay_font_border_thickness: int
    :var help_overlay_background_color: RGB color triple used for the overlay background.
    :vartype help_overlay_background_color: RGBInt
    :var help_overlay_background_alpha: Alpha value for the help overlay background in the range 0 to 255.
    :vartype help_overlay_background_alpha: int
    :var active_head_highlight_color: RGB color triple used for the active head highlight in the human viewer.
    :vartype active_head_highlight_color: RGBInt
    :var active_head_highlight_thickness: Thickness in pixels for the active head highlight.
    :vartype active_head_highlight_thickness: int
    :var cursor_highlight_thickness: Thickness in pixels for cursor highlights.
    :vartype cursor_highlight_thickness: int
    :var cursor_endpoint_highlight_color: RGB color triple used for the cursor endpoint highlight.
    :vartype cursor_endpoint_highlight_color: RGBInt
    :var print_text_in_human_mode: If ``True``, print the text representation of the grid to standard output when
        render mode is ``'human'``. If ``False``, text output is only returned but not printed.
    :vartype print_text_in_human_mode: bool

    .. seealso::
       :mod:`numberlink.render_utils`
          Helpers that convert board state to RGB arrays using these settings.
    """

    endpoint_border_thickness: int = 1
    endpoint_border_color: RGBInt = (255, 255, 255)
    connection_color_adjustment: int = 0
    render_height: int | None = None
    render_width: int | None = None
    # Grid appearance
    grid_background_color: RGBInt = (0, 0, 0)
    gridline_color: RGBInt | None = (128, 128, 128)
    gridline_thickness: int = 1
    # Endpoint number labels
    show_endpoint_numbers: bool = True
    number_font_color: RGBInt = (255, 255, 255)
    number_font_border_color: RGBInt = (0, 0, 0)
    # Thickness in pixels for the font outline drawn around endpoint numbers
    number_font_border_thickness: int = 0
    # Endpoint label font size limits (integer bitmap scale units). None allows automatic sizing based on cell pixels.
    number_font_min_scale: int = 1
    number_font_max_scale: int | None = None
    # Help overlay appearance (human viewer)
    help_overlay_font_color: RGBInt = (220, 220, 220)
    help_overlay_font_border_color: RGBInt = (0, 0, 0)
    # Thickness in pixels for the help-overlay font outline
    help_overlay_font_border_thickness: int = 0
    help_overlay_background_color: RGBInt = (30, 30, 30)
    help_overlay_background_alpha: int = 200
    # Viewer highlights and cursors (human viewer)
    active_head_highlight_color: RGBInt = (255, 255, 255)
    active_head_highlight_thickness: int = 2
    cursor_highlight_thickness: int = 3
    cursor_endpoint_highlight_color: RGBInt = (220, 80, 80)
    print_text_in_human_mode: bool = False
