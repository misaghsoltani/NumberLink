from __future__ import annotations

from pathlib import Path
from typing import cast

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import pytest

from numberlink import GeneratorConfig, NumberLinkRGBEnv, RenderConfig, VariantConfig
from numberlink.types import ActType

from .helpers import base_render_config, save_gif
from .test_utils import add_frame_border


@pytest.mark.visual
def test_solution_retrieval_and_gif(output_dir: Path) -> None:
    """Generate a solvable env, apply solution actions, and save a GIF."""
    env: NumberLinkRGBEnv = NumberLinkRGBEnv(
        render_mode="rgb_array",
        generator=GeneratorConfig(
            mode="hamiltonian", colors=4, width=8, height=8, must_fill=True, allow_diagonal=False, seed=999
        ),
        variant=VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False),
        render_config=RenderConfig(
            render_height=8 * 10,
            render_width=8 * 10,
            endpoint_border_thickness=1,
            endpoint_border_color=(255, 255, 255),
            gridline_color=(80, 80, 80),
            gridline_thickness=1,
            show_endpoint_numbers=True,
            number_font_min_scale=1,
            number_font_max_scale=None,
        ),
    )

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray) and obs.dtype == np.uint8
    assert obs.dtype == np.uint8 and obs.ndim == 3 and obs.shape[-1] == 3
    assert "action_mask" in info

    solution: list[ActType] | None = env.get_solution()
    assert solution is not None and len(solution) > 0

    frames: list[NDArray[np.uint8]] = [add_frame_border(cast(NDArray[np.uint8], env.render()))]
    for act in solution:
        action_idx = int(act)  # coerce numpy scalar if present
        obs, reward, terminated, truncated, info = env.step(action_idx)
        assert isinstance(obs, np.ndarray) and obs.dtype == np.uint8
        rendered: NDArray[np.uint8] | list[NDArray[np.uint8]] | None = cast(
            NDArray[np.uint8] | list[NDArray[np.uint8]] | None, env.render()
        )
        assert rendered is not None
        frames.append(add_frame_border(rendered))
        if terminated or truncated:
            break

    # duplicate last frame for a small hold
    frames.extend(frames[-1:] * 10)
    save_gif(frames, output_dir / "test_solution.gif", fps=8)
    env.close()


@pytest.mark.parametrize(
    "mode_cfg,variant",
    [
        (GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=1), VariantConfig()),
        (
            GeneratorConfig(mode="random_walk", width=6, height=6, colors=3, allow_diagonal=True, seed=2),
            VariantConfig(allow_diagonal=True),
        ),
        (
            GeneratorConfig(mode="random_walk", width=7, height=7, colors=4, bridges_probability=0.2, seed=3),
            VariantConfig(bridges_enabled=True),
        ),
        (
            GeneratorConfig(mode="hamiltonian", width=6, height=6, colors=4, must_fill=True, seed=4),
            VariantConfig(must_fill=True),
        ),
        # cell switching mode
        (
            GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=5),
            VariantConfig(cell_switching_mode=True),
        ),
    ],
)
def test_env_basic_step(mode_cfg: GeneratorConfig, variant: VariantConfig) -> None:
    """Sanity-check stepping works across all feature flags and both generators."""
    # Render config derived from size for crisp RGB arrays
    # Use None (automatic) for font_max in one case, numeric in another
    rc: RenderConfig = base_render_config(mode_cfg.height, mode_cfg.width, font_max=None)
    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=mode_cfg, variant=variant, render_config=rc)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray) and obs.dtype == np.uint8
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n > 0

    # take a few random valid actions
    mask: NDArray[np.uint8] = cast(NDArray[np.uint8], info["action_mask"])
    for _ in range(10):
        valid = np.where(mask > 0)[0]
        if valid.size == 0:
            break
        a: int = np.random.default_rng().choice(valid)
        obs, reward, terminated, truncated, info = env.step(a)
        assert isinstance(obs, np.ndarray) and obs.dtype == np.uint8
        mask = cast(NDArray[np.uint8], info["action_mask"])
        if terminated or truncated:
            break
    env.close()
