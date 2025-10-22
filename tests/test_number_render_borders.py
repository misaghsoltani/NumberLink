import numpy as np
from numpy.typing import NDArray

from numberlink.number_render import BITMAP_FONT, render_bitmap_text_to_array


def _build_glyph_mask_from_font(ch: str, scale: int) -> NDArray[np.bool_]:
    rows: tuple[int, ...] = BITMAP_FONT.get(ch.upper(), BITMAP_FONT[" "])
    h: int = 7 * scale
    w: int = 5 * scale
    mask: NDArray[np.bool_] = np.zeros((h, w), dtype=bool)
    for ry, rowbits in enumerate(rows):
        if rowbits == 0:
            continue
        for cx in range(5):
            if (rowbits >> (4 - cx)) & 1:
                y0: int = ry * scale
                x0: int = cx * scale
                mask[y0 : y0 + scale, x0 : x0 + scale] = True
    return mask


def _dilate8(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    pad: NDArray[np.bool_] = np.pad(mask, 1, mode="constant", constant_values=False)
    dil: NDArray[np.bool_] = (
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
    return dil


def test_border_exterior_only() -> None:
    """Render a selection of glyphs and assert their borders are exterior-only."""
    chars = ["0", "8", "A", "C"]
    scales = [1, 2, 3]
    thicknesses = [1, 2]

    for ch in chars:
        for scale in scales:
            base_mask = _build_glyph_mask_from_font(ch, scale)

            # union of dilations up to max thickness
            current = base_mask.copy()
            union = np.zeros_like(base_mask)
            for _ in range(1, max(thicknesses) + 1):
                current = _dilate8(current)
                union |= current

            # outside-connected background mask (flood fill)
            outside = ~base_mask
            outside_conn = np.zeros_like(outside)
            outside_conn[0, :] = outside[0, :]
            outside_conn[-1, :] = outside[-1, :]
            outside_conn[:, 0] = outside[:, 0]
            outside_conn[:, -1] = outside[:, -1]
            while True:
                nbr = _dilate8(outside_conn)
                new_out = outside_conn | (nbr & outside)
                if new_out.sum() == outside_conn.sum():
                    break
                outside_conn = new_out

            expected_border = union & outside_conn & (~base_mask)

            for thickness in thicknesses:
                h, w = base_mask.shape
                img = np.zeros((h + 4, w + 4, 3), dtype=np.uint8)
                topleft = (2, 2)

                render_bitmap_text_to_array(
                    ch,
                    topleft,
                    (255, 255, 255),
                    img,
                    scale=scale,
                    outline_color=(255, 0, 0),
                    outline_thickness=thickness,
                )

                region = img[2 : 2 + h, 2 : 2 + w]
                fill_mask = np.all(region == np.array([255, 255, 255]), axis=-1)
                border_mask = np.all(region == np.array([255, 0, 0]), axis=-1)

                assert np.array_equal(fill_mask, base_mask), f"Fill mismatch for {ch} scale={scale}"
                assert np.all(border_mask <= expected_border), (
                    f"Border contains interior pixels for {ch} scale={scale} thickness={thickness}"
                )
