"""Rendering utilities for NumberLinkRGB visualizations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .config import RenderConfig
    from .types import ObsType


def upscale_with_endpoint_borders(
    frame: ObsType, endpoint_mask: NDArray[np.bool_], render_config: RenderConfig, scale: int = 1
) -> ObsType:
    """Upscale a cell-based frame to pixel dimensions and draw endpoint borders.

    The input ``frame`` is an RGB array with one pixel per cell. The function repeats pixels to reach the requested
    ``scale`` and draws borders around cells marked by ``endpoint_mask`` using colors and thickness from
    :class:`RenderConfig`.

    :param frame: RGB image array of shape ``(H, W, 3)`` where each pixel maps to one cell.
    :type frame: ObsType
    :param endpoint_mask: Boolean mask indicating endpoint positions with shape ``(H, W)``.
    :type endpoint_mask: NDArray[bool]
    :param render_config: Rendering configuration specifying border color and thickness.
    :type render_config: RenderConfig
    :param scale: Number of pixels per cell in the output. Must be at least ``1``, when smaller than twice the border
        thickness it is increased to accommodate the border.
    :type scale: int, optional

    :return: Upscaled frame with endpoint borders drawn, shape ``(H*scale, W*scale, 3)``.
    :rtype: ObsType
    """
    if scale < 1:
        raise ValueError(f"Scale must be >= 1, got {scale}")

    thickness: int = render_config.endpoint_border_thickness

    if scale < thickness * 2:
        scale = max(scale, thickness * 2)

    # Create upscaled image by repeating pixels
    upscaled: ObsType = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)

    if thickness <= 0 or not np.any(endpoint_mask):
        return upscaled

    border_color: NDArray[np.uint8] = np.array(render_config.endpoint_border_color, dtype=np.uint8)

    # Draw borders for each endpoint
    ep_positions: NDArray[np.intp] = np.argwhere(endpoint_mask)

    for cell_r, cell_c in ep_positions:
        # Calculate pixel coordinates for this cell
        pixel_r_start: int = cell_r * scale
        pixel_r_end: int = pixel_r_start + scale
        pixel_c_start: int = cell_c * scale
        pixel_c_end: int = pixel_c_start + scale

        # Draw top border
        upscaled[pixel_r_start : pixel_r_start + thickness, pixel_c_start:pixel_c_end] = border_color

        # Draw bottom border
        upscaled[pixel_r_end - thickness : pixel_r_end, pixel_c_start:pixel_c_end] = border_color

        # Draw left border
        upscaled[pixel_r_start:pixel_r_end, pixel_c_start : pixel_c_start + thickness] = border_color

        # Draw right border
        upscaled[pixel_r_start:pixel_r_end, pixel_c_end - thickness : pixel_c_end] = border_color

    return upscaled
