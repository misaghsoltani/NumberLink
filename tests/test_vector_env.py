from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from numberlink import GeneratorConfig, NumberLinkRGBVectorEnv, RenderConfig, VariantConfig

from .helpers import save_gif, standard_scenarios
from .test_utils import add_frame_border, tile_images

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator
    from numpy.typing import NDArray


@pytest.mark.visual
def test_vector_parallel_scenarios(output_dir: Path) -> None:
    """Run several vector envs in parallel and save tiled GIFs."""
    scenarios: list[tuple[str, GeneratorConfig, VariantConfig, bool]] = standard_scenarios()

    envs: list[NumberLinkRGBVectorEnv] = []
    names: list[str] = []
    try:
        for idx, (name, gen, var, show_numbers) in enumerate(scenarios):
            # Alternate between no explicit max and a small numeric max to test both paths
            font_max: int | None = None if (idx % 2 == 0) else 3
            rc = RenderConfig(
                render_height=gen.height * 12,
                render_width=gen.width * 12,
                endpoint_border_thickness=1,
                endpoint_border_color=(255, 255, 255),
                show_endpoint_numbers=show_numbers,
                number_font_min_scale=1,
                number_font_max_scale=font_max,
            )
            envs.append(
                NumberLinkRGBVectorEnv(
                    num_envs=1, render_mode="rgb_array", generator=gen, variant=var, render_config=rc, step_limit=500
                )
            )
            names.append(name)

        # reset
        frames_per: list[list[NDArray[np.uint8]]] = []
        masks: list[NDArray[np.uint8]] = []
        for env in envs:
            obs, info = env.reset()
            frames_per.append([add_frame_border(obs[0].copy())])
            masks.append(cast("NDArray[np.uint8]", info["action_mask"])[0])

        done: list[bool] = [False] * len(envs)
        rng: Generator = np.random.default_rng(0)
        max_steps = 300
        for _ in range(max_steps):
            if all(done):
                break
            for idx, env in enumerate(envs):
                if done[idx]:
                    continue
                # sample a valid action from current mask
                valid = np.where(masks[idx] > 0)[0]
                if valid.size == 0:
                    done[idx] = True
                    continue
                a = int(rng.choice(valid))
                obs, _rewards, terminated, truncated, infos = env.step(np.array([a]))
                masks[idx] = cast("NDArray[np.uint8]", infos["action_mask"])[0]
                frames_per[idx].append(add_frame_border(obs[0].copy()))
                if terminated[0] or truncated[0]:
                    done[idx] = True

        # pad to common length
        max_len: int = max(len(fr) for fr in frames_per)
        for fr in frames_per:
            while len(fr) < max_len:
                fr.append(fr[-1])

        # tile and save combined gif
        rows = math.ceil(math.sqrt(len(frames_per)))
        cols = math.ceil(len(frames_per) / rows)
        combined: list[NDArray[np.uint8]] = [
            tile_images([frames_per[i][t] for i in range(len(frames_per))], (rows, cols)) for t in range(max_len)
        ]
        save_gif(combined, output_dir / "vector_scenarios.gif", fps=10)

        # save per-scenario GIFs
        for i, name in enumerate(names):
            save_gif(frames_per[i], output_dir / f"vector_{name}.gif", fps=10)
    finally:
        for env in envs:
            env.close()
