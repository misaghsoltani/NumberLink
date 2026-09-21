from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from numberlink import GeneratorConfig, NumberLinkRGBEnv, VariantConfig

from .helpers import save_gif

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator
    from numpy.typing import NDArray


def test_cell_switching_encode_decode_roundtrip() -> None:
    """Ensure encode/decode cycle is consistent for cell switching actions."""
    env = NumberLinkRGBEnv(
        render_mode=None,
        generator=GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=7),
        variant=VariantConfig(cell_switching_mode=True),
    )
    _obs, _info = env.reset()
    # pick a non-endpoint cell
    mask: NDArray[np.uint8] = env.compute_action_mask()
    valid = np.where(mask > 0)[0]
    assert valid.size > 0
    a = int(valid[0])
    r, c, color_value = env.decode_cell_switching_action(a)
    assert 0 <= r < env.H
    assert 0 <= c < env.W
    ra: int = env.encode_cell_switching_action(r, c, color_value)
    assert int(ra) == int(a)
    env.close()


@pytest.mark.visual
def test_cell_switching_progress_and_gif(output_dir: Path) -> None:
    """Exercise cell switching with random valid actions and save a GIF."""
    env = NumberLinkRGBEnv(
        render_mode="rgb_array",
        generator=GeneratorConfig(mode="random_walk", width=6, height=6, colors=3, seed=11),
        variant=VariantConfig(cell_switching_mode=True),
    )
    env.reset()
    frames: list[NDArray[np.uint8]] = [env.render_rgb()]
    rng: Generator = np.random.default_rng(0)
    for _ in range(200):
        # randomly paint a non-endpoint cell with a random color
        mask: NDArray[np.uint8] = env.compute_action_mask()
        valid = np.where(mask > 0)[0]
        if valid.size == 0:
            break
        a = int(rng.choice(valid))
        _obs, _reward, terminated, truncated, _info = env.step(a)
        frames.append(env.render_rgb())
        if terminated or truncated:
            break

    # ensure we produced some frames and save gif
    save_gif(frames, output_dir / "cell_switching.gif", fps=10)
    env.close()
