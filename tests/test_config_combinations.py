from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from numberlink import GeneratorConfig, NumberLinkRGBEnv, RenderConfig, RewardConfig, VariantConfig
from numberlink.types import RGBInt

from .helpers import save_gif
from .test_utils import add_frame_border


def small_render_config(h: int, w: int, show_numbers: bool, font_max: int | None = None) -> RenderConfig:
    """Return a compact RenderConfig sized to grid for fast tests."""
    ppc = 10
    return RenderConfig(
        render_height=h * ppc,
        render_width=w * ppc,
        endpoint_border_thickness=1,
        endpoint_border_color=(255, 255, 255),
        gridline_color=(40, 40, 40),
        gridline_thickness=1,
        show_endpoint_numbers=show_numbers,
        number_font_min_scale=1,
        number_font_max_scale=font_max,
    )


def visual_render_config(h: int, w: int, **kwargs: Any) -> RenderConfig:
    """Return a RenderConfig for visual tests with customizable parameters."""
    ppc = 16
    defaults: dict[str, Any] = {
        "render_height": h * ppc,
        "render_width": w * ppc,
        "endpoint_border_thickness": 1,
        "endpoint_border_color": (255, 255, 255),
        "gridline_color": (60, 60, 60),
        "gridline_thickness": 1,
        "show_endpoint_numbers": False,
        "number_font_min_scale": 1,
        "number_font_max_scale": None,
    }
    defaults.update(kwargs)
    return RenderConfig(**defaults)


def run_env_and_capture(env: NumberLinkRGBEnv, max_steps: int = 30) -> tuple[list[np.ndarray], bool]:
    """Run environment taking valid actions and capture frames. Returns (frames, won)."""
    frames: list[np.ndarray] = []
    obs, info = env.reset()
    frames.append(add_frame_border(cast(np.ndarray, env.render())))

    mask: NDArray[np.uint8] = cast(NDArray[np.uint8], info["action_mask"])
    won = False

    for _ in range(max_steps):
        valid = np.where(mask > 0)[0]
        if valid.size == 0:
            break
        act = int(valid[0])
        obs, reward, terminated, truncated, info = env.step(act)
        frames.append(add_frame_border(cast(np.ndarray, env.render())))
        mask = cast(NDArray[np.uint8], info["action_mask"])

        if terminated:
            won = True
            break
        if truncated:
            break

    # Hold last frame
    frames.extend([frames[-1]] * 5)
    return frames, won


@pytest.mark.parametrize(
    "gen_cfg",
    [
        GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=1),
        GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=2),
        GeneratorConfig(mode="random_walk", width=6, height=5, colors=3, bridges_probability=0.3, seed=3),
        GeneratorConfig(mode="hamiltonian", width=6, height=6, colors=4, seed=4),
        GeneratorConfig(mode="hamiltonian", width=5, height=6, colors=3, seed=5),
    ],
)
@pytest.mark.parametrize(
    "variant",
    [
        VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False, cell_switching_mode=False),
        VariantConfig(must_fill=False, allow_diagonal=False, bridges_enabled=False, cell_switching_mode=False),
        VariantConfig(must_fill=True, allow_diagonal=True, bridges_enabled=False, cell_switching_mode=False),
        VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=True, cell_switching_mode=False),
        VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False, cell_switching_mode=True),
    ],
)
@pytest.mark.parametrize("show_numbers", [False, True])
@pytest.mark.parametrize(
    "reward",
    [
        RewardConfig(step_penalty=-0.01, invalid_penalty=-0.05, connect_bonus=0.5, win_bonus=5.0),
        RewardConfig(step_penalty=-0.001, invalid_penalty=-0.1, connect_bonus=0.25, win_bonus=3.0),
    ],
)
def test_all_config_combinations_smoke(
    gen_cfg: GeneratorConfig, variant: VariantConfig, show_numbers: bool, reward: RewardConfig
) -> None:
    """Smoke test stepping across cross-product of Generator/Variant/Render/Reward configs."""
    variant = VariantConfig(
        must_fill=variant.must_fill,
        allow_diagonal=variant.allow_diagonal,
        bridges_enabled=variant.bridges_enabled or (gen_cfg.bridges_probability > 0.0),
        cell_switching_mode=variant.cell_switching_mode,
    )

    # Exercise both automatic (None) and bounded (small numeric) max-scale configurations in the same smoke test
    for font_max in (None, 3):
        rc: RenderConfig = small_render_config(gen_cfg.height, gen_cfg.width, show_numbers, font_max=font_max)
    # Limit steps to keep runtime low
    env = NumberLinkRGBEnv(
        render_mode="rgb_array",
        generator=gen_cfg,
        variant=variant,
        reward_config=reward,
        render_config=rc,
        step_limit=gen_cfg.height * gen_cfg.width * 2,
    )
    try:
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.uint8
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert "action_mask" in info

        # Take up to 15 valid actions (or until done)
        mask: NDArray[np.uint8] = cast(NDArray[np.uint8], info["action_mask"])
        steps = 0
        while steps < 15:
            valid = np.where(mask > 0)[0]
            if valid.size == 0:
                break
            act = int(valid[0])
            obs, reward_val, terminated, truncated, info = env.step(act)
            assert isinstance(obs, np.ndarray)
            assert obs.dtype == np.uint8
            mask = cast(NDArray[np.uint8], info["action_mask"])
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()


# VISUAL TESTS - VariantConfig combinations


@pytest.mark.visual
@pytest.mark.parametrize(
    "variant_name,variant,gen_overrides",
    [
        (
            "standard",
            VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False, cell_switching_mode=False),
            {},
        ),
        (
            "no_must_fill",
            VariantConfig(must_fill=False, allow_diagonal=False, bridges_enabled=False, cell_switching_mode=False),
            {},
        ),
        (
            "diagonal",
            VariantConfig(must_fill=True, allow_diagonal=True, bridges_enabled=False, cell_switching_mode=False),
            {},
        ),
        (
            "bridges",
            VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=True, cell_switching_mode=False),
            {"mode": "random_walk", "bridges_probability": 0.3},
        ),
        (
            "cell_switching",
            VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False, cell_switching_mode=True),
            {},
        ),
        (
            "diagonal_no_fill",
            VariantConfig(must_fill=False, allow_diagonal=True, bridges_enabled=False, cell_switching_mode=False),
            {},
        ),
        (
            "bridges_diagonal",
            VariantConfig(must_fill=True, allow_diagonal=True, bridges_enabled=True, cell_switching_mode=False),
            {"mode": "random_walk", "bridges_probability": 0.3},
        ),
        (
            "cell_switching_diagonal",
            VariantConfig(must_fill=True, allow_diagonal=True, bridges_enabled=False, cell_switching_mode=True),
            {},
        ),
    ],
)
def test_visual_variant_configs(
    output_dir: Path, variant_name: str, variant: VariantConfig, gen_overrides: dict[str, Any]
) -> None:
    """Visual test for each VariantConfig combination."""
    gen = GeneratorConfig(width=6, height=6, colors=3, seed=100, **gen_overrides)
    render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)

    env = NumberLinkRGBEnv(
        render_mode="rgb_array", generator=gen, variant=variant, render_config=render, step_limit=100
    )

    try:
        frames, won = run_env_and_capture(env, max_steps=40)
        save_gif(frames, output_dir / f"variant_{variant_name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


# VISUAL TESTS - GeneratorConfig modes and parameters


@pytest.mark.visual
@pytest.mark.parametrize(
    "mode,seed,name",
    [
        ("random_walk", 42, "random_walk_1"),
        ("random_walk", 123, "random_walk_2"),
        ("hamiltonian", 42, "hamiltonian_1"),
        ("hamiltonian", 456, "hamiltonian_2"),
    ],
)
def test_visual_generator_modes(output_dir: Path, mode: str, seed: int, name: str) -> None:
    """Visual test for different generator modes."""
    gen = GeneratorConfig(mode=mode, width=6, height=6, colors=4, seed=seed)
    render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=40)
        save_gif(frames, output_dir / f"generator_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "width,height,colors,name",
    [
        (4, 4, 2, "small_4x4"),
        (6, 6, 3, "medium_6x6"),
        (8, 8, 5, "large_8x8"),
        (5, 7, 3, "rect_5x7"),
        (8, 5, 4, "rect_8x5"),
    ],
)
def test_visual_generator_sizes(output_dir: Path, width: int, height: int, colors: int, name: str) -> None:
    """Visual test for different board sizes and color counts."""
    gen = GeneratorConfig(mode="random_walk", width=width, height=height, colors=colors, seed=200)
    render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=width * height * 2)

    try:
        frames, won = run_env_and_capture(env, max_steps=50)
        save_gif(frames, output_dir / f"size_{name}.gif", fps=10)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize("bridge_prob,name", [(0.0, "no_bridges"), (0.2, "few_bridges"), (0.5, "many_bridges")])
def test_visual_bridges_probability(output_dir: Path, bridge_prob: float, name: str) -> None:
    """Visual test for different bridge probabilities."""
    gen = GeneratorConfig(mode="random_walk", width=7, height=7, colors=4, bridges_probability=bridge_prob, seed=300)
    variant = VariantConfig(bridges_enabled=(bridge_prob > 0))
    render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)

    env = NumberLinkRGBEnv(
        render_mode="rgb_array", generator=gen, variant=variant, render_config=render, step_limit=150
    )

    try:
        frames, won = run_env_and_capture(env, max_steps=50)
        save_gif(frames, output_dir / f"bridges_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


# VISUAL TESTS - RenderConfig visual parameters


@pytest.mark.visual
@pytest.mark.parametrize(
    "border_thick,border_color,name",
    [
        (0, (255, 255, 255), "no_border"),
        (1, (255, 255, 255), "thin_white"),
        (2, (255, 255, 255), "thick_white"),
        (1, (255, 0, 0), "thin_red"),
        (2, (0, 255, 0), "thick_green"),
        (3, (0, 0, 255), "thicker_blue"),
    ],
)
def test_visual_endpoint_borders(output_dir: Path, border_thick: int, border_color: RGBInt, name: str) -> None:
    """Visual test for different endpoint border thicknesses and colors."""
    gen = GeneratorConfig(width=6, height=6, colors=3, seed=400)
    render: RenderConfig = visual_render_config(
        gen.height,
        gen.width,
        endpoint_border_thickness=border_thick,
        endpoint_border_color=border_color,
        show_endpoint_numbers=True,
    )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"endpoint_border_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "gridline_color,gridline_thick,name",
    [
        (None, 1, "no_gridlines"),
        ((60, 60, 60), 1, "gray_thin"),
        ((100, 100, 100), 1, "lightgray_thin"),
        ((255, 255, 255), 1, "white_thin"),
        ((60, 60, 60), 2, "gray_thick"),
        ((255, 0, 0), 2, "red_thick"),
    ],
)
def test_visual_gridlines(output_dir: Path, gridline_color: RGBInt | None, gridline_thick: int, name: str) -> None:
    """Visual test for different gridline colors and thicknesses."""
    gen = GeneratorConfig(width=6, height=6, colors=3, seed=500)
    render: RenderConfig = visual_render_config(
        gen.height,
        gen.width,
        gridline_color=gridline_color,
        gridline_thickness=gridline_thick,
        show_endpoint_numbers=True,
    )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"gridlines_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "bg_color,name",
    [
        ((0, 0, 0), "black"),
        ((255, 255, 255), "white"),
        ((30, 30, 60), "dark_blue"),
        ((60, 30, 30), "dark_red"),
        ((20, 40, 20), "dark_green"),
    ],
)
def test_visual_background_colors(output_dir: Path, bg_color: RGBInt, name: str) -> None:
    """Visual test for different background colors."""
    gen = GeneratorConfig(width=6, height=6, colors=3, seed=600)
    render: RenderConfig = visual_render_config(
        gen.height, gen.width, grid_background_color=bg_color, show_endpoint_numbers=True
    )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"background_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "show_numbers,font_color,font_border_color,font_border_thick,name",
    [
        (True, (255, 255, 255), (0, 0, 0), 1, "white_black_thin"),
        (True, (0, 0, 0), (255, 255, 255), 1, "black_white_thin"),
        (True, (255, 255, 255), (0, 0, 0), 2, "white_black_thick"),
        (True, (255, 255, 0), (0, 0, 255), 2, "yellow_blue_thick"),
        (True, (0, 255, 0), (255, 0, 255), 1, "green_magenta"),
        (False, (255, 255, 255), (0, 0, 0), 1, "no_numbers"),
    ],
)
def test_visual_endpoint_numbers(
    output_dir: Path,
    show_numbers: bool,
    font_color: RGBInt,
    font_border_color: RGBInt,
    font_border_thick: int,
    name: str,
) -> None:
    """Visual test for endpoint number rendering styles."""
    gen = GeneratorConfig(width=6, height=6, colors=4, seed=700)
    render: RenderConfig = visual_render_config(
        gen.height,
        gen.width,
        show_endpoint_numbers=show_numbers,
        number_font_color=font_color,
        number_font_border_color=font_border_color,
        number_font_border_thickness=font_border_thick,
    )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"numbers_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "font_min,font_max,name",
    [(1, None, "auto_scale"), (1, 2, "limited_small"), (1, 3, "limited_medium"), (2, 4, "limited_large")],
)
def test_visual_font_scaling(output_dir: Path, font_min: int, font_max: int | None, name: str) -> None:
    """Visual test for font min/max scaling parameters."""
    gen = GeneratorConfig(width=6, height=6, colors=4, seed=800)
    render: RenderConfig = visual_render_config(
        gen.height,
        gen.width,
        show_endpoint_numbers=True,
        number_font_min_scale=font_min,
        number_font_max_scale=font_max,
    )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"font_scale_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "adjustment,name",
    [(-50, "darker"), (-20, "slightly_darker"), (0, "normal"), (20, "slightly_brighter"), (50, "brighter")],
)
def test_visual_connection_color_adjustment(output_dir: Path, adjustment: int, name: str) -> None:
    """Visual test for connection color adjustment."""
    gen = GeneratorConfig(width=6, height=6, colors=3, seed=900)
    render: RenderConfig = visual_render_config(
        gen.height,
        gen.width,
        connection_color_adjustment=adjustment,
        endpoint_border_thickness=0,  # So connections are visible
        show_endpoint_numbers=True,
    )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"color_adjustment_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize(
    "render_h,render_w,name",
    [(None, None, "default"), (120, 120, "120x120"), (200, 200, "200x200"), (96, 160, "96x160_rect")],
)
def test_visual_render_dimensions(output_dir: Path, render_h: int | None, render_w: int | None, name: str) -> None:
    """Visual test for different render dimensions."""
    gen = GeneratorConfig(width=6, height=6, colors=3, seed=1000)

    if render_h is None and render_w is None:
        render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)
    else:
        render = RenderConfig(
            render_height=render_h,
            render_width=render_w,
            endpoint_border_thickness=1,
            endpoint_border_color=(255, 255, 255),
            gridline_color=(60, 60, 60),
            gridline_thickness=1,
            show_endpoint_numbers=True,
        )

    env = NumberLinkRGBEnv(render_mode="rgb_array", generator=gen, render_config=render, step_limit=100)

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"dimensions_{name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


# VISUAL TESTS - RewardConfig combinations (affects gameplay)


@pytest.mark.visual
@pytest.mark.parametrize(
    "reward_name,reward",
    [
        ("default", RewardConfig()),
        ("high_penalties", RewardConfig(step_penalty=-0.1, invalid_penalty=-0.5)),
        ("high_rewards", RewardConfig(connect_bonus=2.0, win_bonus=10.0)),
        ("minimal", RewardConfig(step_penalty=-0.001, invalid_penalty=-0.01, connect_bonus=0.1, win_bonus=1.0)),
        ("balanced", RewardConfig(step_penalty=-0.05, invalid_penalty=-0.1, connect_bonus=1.0, win_bonus=5.0)),
    ],
)
def test_visual_reward_configs(output_dir: Path, reward_name: str, reward: RewardConfig) -> None:
    """Visual test for different reward configurations (gameplay behavior test)."""
    gen = GeneratorConfig(width=5, height=5, colors=2, seed=1100)
    render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)

    env = NumberLinkRGBEnv(
        render_mode="rgb_array", generator=gen, reward_config=reward, render_config=render, step_limit=80
    )

    try:
        frames, won = run_env_and_capture(env, max_steps=30)
        save_gif(frames, output_dir / f"reward_{reward_name}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()


# VISUAL TESTS - Complex combinations


@pytest.mark.visual
def test_visual_all_features_combined(output_dir: Path) -> None:
    """Visual test combining multiple advanced features."""
    gen = GeneratorConfig(mode="random_walk", width=8, height=8, colors=5, bridges_probability=0.2, seed=1200)
    variant = VariantConfig(must_fill=True, allow_diagonal=True, bridges_enabled=True, cell_switching_mode=False)
    render: RenderConfig = visual_render_config(
        gen.height,
        gen.width,
        show_endpoint_numbers=True,
        endpoint_border_thickness=2,
        endpoint_border_color=(255, 255, 0),
        gridline_color=(100, 100, 100),
        gridline_thickness=2,
        number_font_color=(255, 255, 255),
        number_font_border_color=(0, 0, 0),
        number_font_border_thickness=2,
    )
    reward = RewardConfig(step_penalty=-0.02, connect_bonus=1.0, win_bonus=10.0)

    env = NumberLinkRGBEnv(
        render_mode="rgb_array",
        generator=gen,
        variant=variant,
        reward_config=reward,
        render_config=render,
        step_limit=200,
    )

    try:
        frames, won = run_env_and_capture(env, max_steps=50)
        save_gif(frames, output_dir / "combined_all_features.gif", fps=10)
        assert len(frames) > 0
    finally:
        env.close()


@pytest.mark.visual
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_visual_variety_showcase(output_dir: Path, seed: int) -> None:
    """Visual test showcasing variety in generated puzzles."""
    # Determine mode based on seed
    mode: str = "random_walk" if seed % 2 == 0 else "hamiltonian"
    # Only enable bridges for random_walk mode (hamiltonian doesn't support bridges)
    use_bridges: bool = (seed % 2 == 0) and (seed > 2)
    allow_diag: bool = seed % 3 == 0

    gen = GeneratorConfig(
        mode=mode, width=7, height=7, colors=4, bridges_probability=0.2 if use_bridges else 0.0, seed=1300 + seed
    )
    variant = VariantConfig(
        must_fill=True,
        allow_diagonal=allow_diag,
        bridges_enabled=(gen.bridges_probability > 0),
        cell_switching_mode=False,
    )
    render: RenderConfig = visual_render_config(gen.height, gen.width, show_endpoint_numbers=True)

    env = NumberLinkRGBEnv(
        render_mode="rgb_array", generator=gen, variant=variant, render_config=render, step_limit=150
    )

    try:
        frames, won = run_env_and_capture(env, max_steps=40)
        save_gif(frames, output_dir / f"variety_seed_{seed}.gif", fps=8)
        assert len(frames) > 0
    finally:
        env.close()
