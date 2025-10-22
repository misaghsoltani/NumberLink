"""Bitmap text rendering utilities.

This module provides lightweight bitmap font data and helper functions to render small text overlays into RGB image
arrays. The functions are intended for debug rendering and simple user interfaces where external font dependencies are
undesirable. Use :func:`render_bitmap_text_to_array` to draw left-aligned text, :func:`render_bitmap_text_centered` to
draw centered text, and :func:`build_endpoint_labels` to compute label positions for grid endpoints.

The module exposes a compact bitmap font mapping in :data:`BITMAP_FONT`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from .types import Coord, RGBInt


# Tiny bitmap font for rendering endpoint numbers and text overlays
BITMAP_FONT: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0, 0, 0),
    "0": (0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110),
    "1": (0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "2": (0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111),
    "3": (0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110),
    "4": (0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010),
    "5": (0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110),
    "6": (0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110),
    "7": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000),
    "8": (0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110),
    "9": (0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100),
    "A": (0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "B": (0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110),
    "C": (0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110),
    "D": (0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100),
    "E": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111),
    "F": (0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000),
    "G": (0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110),
    "H": (0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001),
    "I": (0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110),
    "J": (0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100),
    "K": (0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001),
    "L": (0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111),
    "M": (0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001),
    "N": (0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001),
    "O": (0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "P": (0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000),
    "Q": (0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101),
    "R": (0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001),
    "S": (0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110),
    "T": (0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100),
    "U": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110),
    "V": (0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100),
    "W": (0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001),
    "X": (0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001),
    "Y": (0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100),
    "Z": (0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111),
}


def render_bitmap_text_to_array(
    text: str,
    topleft: Coord,
    color: RGBInt,
    img: NDArray[np.uint8],
    scale: int = 2,
    outline_color: RGBInt | None = (0, 0, 0),
    outline_thickness: int = 1,
) -> int:
    """Render ``text`` into an RGB image array at a top-left position.

    Draw a string using the :data:`BITMAP_FONT` mapping into the provided image array. The function writes directly into
    ``img`` and returns the horizontal advance in pixels from the original ``topleft`` to the end of the rendered text.

    The image array uses the (row, column, channel) indexing convention. The
    ``topleft`` argument uses an (x, y) coordinate convention where ``x`` is
    the column and ``y`` is the row. Use :func:`render_bitmap_text_centered`
    to draw text centered at a point instead of at a top-left corner.

    :param text: Text to render. Characters not present in :data:`BITMAP_FONT` are rendered as space characters.
    :type text: str
    :param topleft: Pixel coordinates for the top-left corner as ``(x, y)`` where ``x`` is column and ``y`` is row.
    :type topleft: :class:`tuple`
    :param color: RGB color tuple with values in the range ``0`` to ``255``.
    :type color: tuple[int, int, int]
    :param img: Target RGB array with shape ``(H, W, 3)`` and dtype ``numpy.uint8``. The array is modified in place.
    :type img: :class:`numpy.ndarray`
    :param scale: Pixel size used for each bitmap cell. Min value is ``1``. Larger values scale the glyphs uniformly.
        Defaults to ``2``.
    :type scale: int, optional
    :param outline_color: RGB tuple used to draw an outline around glyphs. If ``None``, no outline is drawn. Defaults to
        ``(0, 0, 0)``.
    :type outline_color: tuple[int, int, int] or None, optional
    :param outline_thickness: Outline thickness in pixels measured in bitmap units. Values less than ``1`` disable
        outline drawing. Defaults to ``1``.
    :type outline_thickness: int, optional

    :return: Horizontal width in pixels of the rendered string.
    :rtype: int

    .. seealso::
       :func:`render_bitmap_text_centered` for centered text rendering.

    .. note::
       The function modifies ``img`` in place and performs clipping when the glyphs fall outside the image bounds.
    """
    x0, y0 = topleft
    x: int = x0
    y: int = y0
    img_h: int
    img_w: int
    img_h, img_w = img.shape[:2]

    scale = max(scale, 1)

    color_arr: NDArray[np.uint8] = np.asarray(color, dtype=np.uint8)
    outline_arr: NDArray[np.uint8] | None = (
        np.asarray(outline_color, dtype=np.uint8) if outline_color is not None else None
    )

    # Render per-glyph using a mask to compute a single outer border (dilation - mask)
    advance_px: int = (5 + 1) * scale
    for ch in text:
        ch_u: str = ch.upper()
        rows: tuple[int, ...] = BITMAP_FONT.get(ch_u, BITMAP_FONT[" "])
        glyph_h: int = 7 * scale
        glyph_w: int = 5 * scale

        # Compute target window and clip to image bounds early
        x1: int = x + glyph_w
        y1: int = y + glyph_h
        if x >= img_w or y >= img_h or x1 <= 0 or y1 <= 0:
            x += advance_px
            continue

        # Build glyph mask (boolean)
        glyph_mask: NDArray[np.bool_] = np.zeros((glyph_h, glyph_w), dtype=np.bool_)
        for ry, rowbits in enumerate(rows):
            if rowbits == 0:
                continue

            # base 5-bit row expanded to scale in both axes
            for cx in range(5):
                if (rowbits >> (4 - cx)) & 1:
                    y0s: int = ry * scale
                    x0s: int = cx * scale
                    glyph_mask[y0s : y0s + scale, x0s : x0s + scale] = True

        # Compute outer border: dilation using 8-neighborhood minus the mask
        # Support outline thickness > 1 by repeated dilation via padding and logical ORs
        border_mask: NDArray[np.bool_]
        if outline_arr is not None and glyph_mask.any() and outline_thickness > 0:
            # Start with the original mask and dilate outline_thickness times
            current: NDArray[np.bool_] = glyph_mask.copy()
            border_mask = np.zeros_like(glyph_mask)
            # Union dilations for all layers 1..outline_thickness so intermediate layers  dare included
            # A single glyph segment may require multiple border pixels at different distances from the glyph core
            union_dil: NDArray[np.bool_] = np.zeros_like(glyph_mask)
            for _ in range(max(1, int(outline_thickness))):
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

            # Compute outside-connected background mask (flood fill from glyph periphery)
            outside: NDArray[np.bool_] = ~glyph_mask
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

            border_mask = union_dil & outside_conn & (~glyph_mask)
        else:
            border_mask = np.zeros_like(glyph_mask)

        # Compute clipped destination slice once
        dx0: int = max(0, x)
        dy0: int = max(0, y)
        dx1: int = min(img_w, x1)
        dy1: int = min(img_h, y1)
        if dx0 < dx1 and dy0 < dy1:
            sx0: int = dx0 - x
            sy0: int = dy0 - y
            sx1: int = sx0 + (dx1 - dx0)
            sy1: int = sy0 + (dy1 - dy0)
            if outline_arr is not None and outline_thickness > 0:
                bm: NDArray[np.bool_] = border_mask[sy0:sy1, sx0:sx1]
                if bm.any():
                    img[dy0:dy1, dx0:dx1][bm] = outline_arr

            gm: NDArray[np.bool_] = glyph_mask[sy0:sy1, sx0:sx1]
            if gm.any():
                img[dy0:dy1, dx0:dx1][gm] = color_arr

        x += advance_px

    return x - x0


def render_bitmap_text_centered(
    text: str,
    center: Coord,
    color: RGBInt,
    img: NDArray[np.uint8],
    scale: int = 2,
    outline_color: RGBInt | None = (0, 0, 0),
    outline_thickness: int = 1,
) -> None:
    """Render ``text`` centered at the given pixel coordinates.

    Compute a top-left origin that centers the rendered text at ``center`` and delegate to
    :func:`render_bitmap_text_to_array` for actual drawing. The ``center`` argument uses an (x, y) convention where
    ``x`` is column and ``y`` is row.

    :param text: Text to render.
    :type text: str
    :param center: Pixel coordinates for the text center as ``(x, y)`` where ``x`` is column and ``y`` is row.
    :type center: :class:`tuple`
    :param color: RGB color tuple with values in the range ``0`` to ``255``.
    :type color: tuple[int, int, int]
    :param img: Target RGB array with shape ``(H, W, 3)`` and dtype ``numpy.uint8``. The array is modified in place.
    :type img: :class:`numpy.ndarray`
    :param scale: Pixel size used for each bitmap cell. Defaults to ``2``.
    :type scale: int, optional
    :param outline_color: RGB tuple used to draw an outline around glyphs or ``None`` to disable outlining.
        Defaults to ``(0, 0, 0)``.
    :type outline_color: tuple[int, int, int] or None, optional
    :param outline_thickness: Outline thickness in pixels measured in bitmap units. Defaults to ``1``.
    :type outline_thickness: int, optional

    .. seealso::
       :func:`render_bitmap_text_to_array` for the drawing implementation.
    """
    effective_scale: int = max(scale, 1)
    glyph_width: int = 5 * effective_scale
    spacing: int = effective_scale
    if text:
        n_chars: int = len(text)
        width: int = glyph_width * n_chars + spacing * max(n_chars - 1, 0)
    else:
        width = glyph_width
    height: int = 7 * effective_scale
    # Calculate top-left corner from center (both in (x, y) = (column, row) format)
    topleft: Coord = (center[0] - width // 2, center[1] - height // 2)
    render_bitmap_text_to_array(text, topleft, color, img, effective_scale, outline_color, outline_thickness)


def build_endpoint_labels(
    endpoints: Iterable[tuple[Coord, Coord]] | NDArray[np.int_],
    pixels_per_cell_h: int,
    pixels_per_cell_w: int,
    min_scale: int | None = None,
    max_scale: int | None = None,
    gridline_thickness: int = 0,
) -> list[tuple[str, Coord, RGBInt, int]]:
    """Compute label specifications for grid endpoints.

    Accept either a NumPy array of shape ``(num_colors, 2, 2)`` with endpoint coordinates in ``(row, col)`` format or an
    iterable of ``((r0, c0), (r1, c1))`` pairs. For each endpoint this function computes a text label string, a pixel
    center position suitable for :func:`render_bitmap_text_to_array`, and an adaptive bitmap scale that fits within the
    cell size.

    The returned color tuple is a suggested foreground color for the text. The
    caller may override this when drawing. The function does not perform any
    drawing itself.

    :param endpoints: Endpoint coordinate pairs. Supported formats are an array with shape ``(num_colors, 2, 2)`` or an
        iterable of pairs.
    :type endpoints: collections.abc.Iterable or :class:`numpy.ndarray`
    :param pixels_per_cell_h: Height of a single grid cell in pixels.
    :type pixels_per_cell_h: int
    :param pixels_per_cell_w: Width of a single grid cell in pixels.
    :type pixels_per_cell_w: int
    :param min_scale: Minimum bitmap scale to suggest. If ``None`` no minimum is enforced. Defaults to ``None``.
    :type min_scale: int or None, optional
    :param max_scale: Maximum bitmap scale to suggest. If ``None`` no maximum is enforced. Defaults to ``None``.
    :type max_scale: int or None, optional
    :param gridline_thickness: Thickness of grid lines in pixels. Offsets label centers away from gridline pixels.
        Defaults to ``0``.
    :type gridline_thickness: int, optional

    :return: A list of tuples ``(text, center, color, scale)`` where ``text`` is the string to draw, ``center`` is the
        pixel ``(x, y)`` center for :func:`render_bitmap_text_centered`, ``color`` is a suggested RGB foreground tuple,
        and ``scale`` is an integer bitmap scale.
    :rtype: list[tuple[str, tuple[int, int], tuple[int, int, int], int]]

    .. seealso::
       :func:`render_bitmap_text_centered` and :func:`render_bitmap_text_to_array` for drawing functions that consume
       these specifications.
    """
    specs: list[tuple[str, Coord, RGBInt, int]] = []
    eps: NDArray[np.int_] = np.asarray(endpoints)
    num_colors: int = eps.shape[0]

    for ci in range(num_colors):
        # Extract endpoints robustly whether eps is array or nested lists
        # eps format is [color_idx, endpoint_idx, (row, col)]
        ep0: Coord = (int(eps[ci, 0, 0]), int(eps[ci, 0, 1]))  # (row, col)
        ep1: Coord = (int(eps[ci, 1, 0]), int(eps[ci, 1, 1]))  # (row, col)

        # Always use white text. Renderer can overlay a thin black border for readability
        text_color: RGBInt = (255, 255, 255)
        number_text: str = str(ci)

        # Adaptive bitmap scale limited to roughly one third of the cell.
        n_chars: int = max(1, len(number_text))
        glyph_units: int = n_chars * 5 + max(n_chars - 1, 0)
        inner_h: int = max(1, pixels_per_cell_h - gridline_thickness)
        inner_w: int = max(1, pixels_per_cell_w - gridline_thickness)
        third_h_pixels: int = max(1, inner_h // 3)
        third_w_pixels: int = max(1, inner_w // 3)
        limit_h: int = max(1, third_h_pixels // 7)
        limit_w: int = max(1, third_w_pixels // max(1, glyph_units))
        scale_limit: int = max(1, min(limit_h, limit_w))
        scale: int = scale_limit

        # Clamp to configuration limits if provided
        if min_scale is not None:
            scale = max(int(min_scale), scale)
        if max_scale is not None:
            scale = min(int(max_scale), scale)
        scale = max(1, min(scale, scale_limit))

        for ep in (ep0, ep1):
            # Convert grid coordinates (row, col) to pixel coordinates (x, y) = (column, row)
            cell_w: int = pixels_per_cell_w
            cell_h: int = pixels_per_cell_h
            if gridline_thickness > 0:
                gl: int = gridline_thickness
                left_inset: int = (gl + 1) // 2
                right_inset: int = (gl) // 2
                top_inset: int = (gl + 1) // 2
                bottom_inset: int = (gl) // 2

                # Compute interior width/height after subtracting per-side insets but ensure at least 1px
                interior_w: int = max(1, cell_w - (left_inset + right_inset))
                interior_h: int = max(1, cell_h - (top_inset + bottom_inset))

                cx: int = ep[1] * cell_w + left_inset + interior_w // 2
                cy: int = ep[0] * cell_h + top_inset + interior_h // 2
            else:
                cx = ep[1] * cell_w + cell_w // 2
                cy = ep[0] * cell_h + cell_h // 2

            specs.append((number_text, (cx, cy), text_color, scale))

    return specs
