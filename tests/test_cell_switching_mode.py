from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
import pytest

from numberlink import GeneratorConfig, NumberLinkRGBEnv, VariantConfig

from .helpers import save_gif


def test_cell_switching_encode_decode_roundtrip() -> None:
    """Ensure encode/decode cycle is consistent for cell switching actions."""
    env = NumberLinkRGBEnv(
        render_mode=None,
        generator=GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=7),
        variant=VariantConfig(cell_switching_mode=True),
    )
    obs, info = env.reset()
    # pick a non-endpoint cell
    mask: NDArray[np.uint8] = env._compute_action_mask()  # noqa: SLF001 - internal test access
    valid = np.where(mask > 0)[0]
    assert valid.size > 0
    a = int(valid[0])
    r, c, color_value = env._decode_cell_switching_action(a)  # noqa: SLF001
    assert 0 <= r < env.H and 0 <= c < env.W
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
    obs, info = env.reset()
    frames: list[NDArray[np.uint8]] = [env.render()]  # type: ignore[list-item]
    rng: Generator = np.random.default_rng(0)
    for _ in range(200):
        # randomly paint a non-endpoint cell with a random color
        mask: NDArray[np.uint8] = env._compute_action_mask()  # noqa: SLF001
        valid = np.where(mask > 0)[0]
        if valid.size == 0:
            break
        a = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(a)
        frames.append(env.render())  # type: ignore[arg-type]
        if terminated or truncated:
            break

    # ensure we produced some frames and save gif
    frames = [f for f in frames if f is not None]
    save_gif(frames, output_dir / "cell_switching.gif", fps=10)
    env.close()
