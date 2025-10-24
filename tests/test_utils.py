from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from numberlink.types import RGBInt


def add_frame_border(
    image: NDArray[np.uint8] | Sequence[NDArray[np.uint8]], border_px: int = 4, color: RGBInt = (255, 255, 255)
) -> NDArray[np.uint8]:
    """Return a copy of image with a solid border around it.

    Accept either a single HxWx3 uint8 array, or a non-empty sequence where the first element is used.
    """
    if isinstance(image, np.ndarray):
        arr: NDArray[np.uint8] = image
    else:
        # a Python sequence (list/tuple) of images
        if len(image) == 0:
            raise ValueError("Empty image list provided")
        arr = image[0]

    h, w = arr.shape[:2]
    canvas: NDArray[np.uint8] = np.zeros((h + 2 * border_px, w + 2 * border_px, 3), dtype=np.uint8)
    canvas[:, :] = color
    canvas[border_px : border_px + h, border_px : border_px + w] = arr
    return canvas


def tile_images(images: Sequence[NDArray[np.uint8]], grid_shape: tuple[int, int] | None = None) -> NDArray[np.uint8]:
    """Tile images into a near-square grid (or provided grid)."""
    if not images:
        raise ValueError("No images provided to tile")

    n: int = len(images)
    if grid_shape is None:
        # pick near-square grid
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
    else:
        rows, cols = grid_shape

    max_h: int = max(img.shape[0] for img in images)
    max_w: int = max(img.shape[1] for img in images)
    canvas: NDArray[np.uint8] = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

    for idx, img in enumerate(images):
        r: int = idx // cols
        c: int = idx % cols
        y0: int = r * max_h + (max_h - img.shape[0]) // 2
        x0: int = c * max_w + (max_w - img.shape[1]) // 2
        canvas[y0 : y0 + img.shape[0], x0 : x0 + img.shape[1]] = img

    return canvas
