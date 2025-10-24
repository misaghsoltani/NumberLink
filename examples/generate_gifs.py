"""Batch GIF exporter for NumberLink configurations.

This script iterates over the Cartesian product of gameplay variant toggles and supported generator algorithms, captures
RGB frames while replaying the optimal solution returned by ``NumberLinkRGBEnv.get_solution()``, and stores a
high-resolution GIF for each configuration.
"""

from __future__ import annotations

from argparse import ArgumentParser
import itertools
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import imageio.v3 as iio3
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from numberlink.config import GeneratorConfig, RenderConfig, VariantConfig
from numberlink.env import NumberLinkRGBEnv

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from numberlink.env import InfoDict
    from numberlink.types import ActType


REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PACKAGE_NAME: str = "numberlink"


def _variant_combinations() -> Iterable[tuple[str, dict[str, bool]]]:
    """Yield (label, kwargs) pairs for every combination of VariantConfig toggles.

    :yields: Tuple of label and kwargs for constructing a :class:`VariantConfig`.
    :rtype: Iterable[tuple[str, dict[str, bool]]]
    """
    flag_values: list[tuple[str, tuple[bool, bool]]] = [
        ("must_fill", (False, True)),
        ("allow_diagonal", (False, True)),
        ("bridges_enabled", (False, True)),
        ("cell_switching_mode", (False, True)),
    ]
    names: tuple[str, ...] = tuple(name for name, _ in flag_values)
    value_sets: tuple[tuple[bool, bool], ...] = tuple(values for _, values in flag_values)

    for combo in itertools.product(*value_sets):
        kwargs: dict[str, bool] = dict(zip(names, combo, strict=True))
        label: str = "_".join(f"{key}-{int(value)}" for key, value in kwargs.items())
        yield label, kwargs


def build_status_lines(env: NumberLinkRGBEnv, solved: bool, deadlocked: bool, truncated: bool) -> list[str]:
    """Compose status lines summarizing the end state for overlay rendering.

    :param env: Environment instance used to extract runtime statistics.
    :type env: Any
    :param solved: Whether the level was solved.
    :type solved: bool
    :param deadlocked: Whether the environment reached a deadlock.
    :type deadlocked: bool
    :param truncated: Whether the episode was truncated due to step limits.
    :type truncated: bool
    :returns: List of status lines to render in an overlay.
    :rtype: list[str]
    """
    header: str
    if solved:
        header = "Solved"
    elif deadlocked:
        header = "Deadlocked"
    elif truncated:
        header = "Step limit reached"
    else:
        header = "Status"

    steps: int = int(getattr(env, "_steps", 0))
    connected: int = int(np.sum(env._closed))
    total_colors: int = int(env.num_colors)
    return [header, f"Steps: {steps}", f"Connected: {connected}/{total_colors}"]


def draw_status_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a semi-transparent status overlay onto the provided RGB frame.

    :param frame: RGB image as a NumPy array (H x W x 3).
    :type frame: numpy.ndarray
    :param lines: Lines of text to draw inside the overlay panel.
    :type lines: list[str]
    :returns: RGB image with the overlay applied.
    :rtype: numpy.ndarray
    """
    if not lines:
        return frame

    base_image: Image.Image = Image.fromarray(frame).convert("RGBA")
    overlay: Image.Image = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(overlay)
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont = ImageFont.load_default()

    padding: int = 20
    line_spacing: int = 10
    widths: list[int] = []
    heights: list[int] = []
    for line in lines:
        bbox: tuple[float, float, float, float] = draw.textbbox((0, 0), line, font=font)
        # textbbox may return float coordinates depending on font metrics
        widths.append(int(bbox[2] - bbox[0]))
        heights.append(int(bbox[3] - bbox[1]))

    max_width: int = max(widths) if widths else 0
    total_height: int = sum(heights)
    spacing_total: int = line_spacing * (len(lines) - 1) if len(lines) > 1 else 0
    panel_width: int = max_width + padding * 2
    panel_height: int = total_height + padding * 2 + spacing_total

    origin_x: int = max(0, (base_image.width - panel_width) // 2)
    origin_y: int = max(0, (base_image.height - panel_height) // 2)
    draw.rectangle(xy=(origin_x, origin_y, origin_x + panel_width, origin_y + panel_height), fill=(0, 0, 0, 200))

    text_y: int = origin_y + padding
    for idx, line in enumerate(lines):
        draw.text((origin_x + padding, text_y), line, font=font, fill=(255, 255, 255, 255))
        text_y += heights[idx] + line_spacing

    combined: Image.Image = Image.alpha_composite(base_image, overlay)
    return np.array(combined.convert("RGB"))


def generate_gifs(
    output_dir: Path,
    seed: int | None = None,
    fps: int = 4,
    size: int | None = None,
    colors: int | None = None,
    final_pause_seconds: float = 2.0,
) -> None:
    """Generate GIFs for every variant/generator combination.

    This iterates over the Cartesian product of variant toggles and supported
    generator modes, constructs a level for each combination, replays the
    optimal solution, captures frames and writes a high-resolution GIF.

    :param output_dir: Directory where generated GIF files will be saved.
    :type output_dir: pathlib.Path
    :param seed: Base random seed used for level generation, per-attempt seeds
        are derived from this value.
    :type seed: int
    :param fps: Playback frames-per-second for the exported GIFs.
    :type fps: int
    :returns: None
    :rtype: None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_modes: tuple[str, ...] = ("random_walk", "hamiltonian")

    for label, variant_kwargs in _variant_combinations():
        variant: VariantConfig = VariantConfig(**variant_kwargs)
        for mode in generator_modes:
            bridges_probability: float = 0.25 if (variant.bridges_enabled and mode != "hamiltonian") else 0.0
            render_config = RenderConfig(
                render_height=512,
                render_width=512,
                endpoint_border_thickness=3,
                grid_background_color=(15, 15, 20),
                gridline_color=(70, 70, 70),
                gridline_thickness=2,
                show_endpoint_numbers=True,
                number_font_color=(255, 255, 255),
                number_font_border_color=(0, 0, 0),
            )

            # Determine base board size (square) and color count. Allow CLI overrides.
            base_size: int = size if size is not None else 6
            if size is None:
                if variant.must_fill:
                    base_size = 7
                if variant.bridges_enabled:
                    base_size = max(base_size, 7)

            base_colors: int = colors if colors is not None else 4
            if colors is None and variant.cell_switching_mode:
                base_colors = 3
            base_min_path: int = 4
            if variant.must_fill or variant.bridges_enabled:
                base_min_path = 3

            attempt_specs: list[tuple[int, int, int]] = [
                (base_size, base_colors, base_min_path),
                (base_size + 1, base_colors, max(2, base_min_path - 1)),
                (base_size + 1, max(3, base_colors - 1), 2),
            ]

            env: NumberLinkRGBEnv | None = None
            last_error: ValueError | None = None
            for attempt_index, (side, color_count, min_path_len) in enumerate(attempt_specs):
                generator: GeneratorConfig = GeneratorConfig(
                    mode=mode,
                    width=side,
                    height=side,
                    colors=max(2, color_count),
                    bridges_probability=bridges_probability,
                    min_path_length=max(2, min_path_len),
                    max_retries=50,
                    seed=(None if seed is None else seed + attempt_index),
                )
                env = NumberLinkRGBEnv(
                    render_mode="rgb_array",
                    generator=generator,
                    variant=variant,
                    render_config=render_config,
                    step_limit=side * side * 4,
                )

                reset_seed: None | int = None if seed is None else seed + attempt_index
                env.reset(seed=reset_seed)
                break

            if env is None:
                print(f"[WARN] Could not build level for {label} | mode={mode}: {last_error}")
                continue

            solution: list[ActType] | None = env.get_solution()
            if not solution:
                print(f"[WARN] No solution available for {label} | mode={mode}, skipping.")
                env.close()
                continue

            frames: list[NDArray[np.uint8]] = [env._render_rgb().copy()]
            final_info: InfoDict | None = None
            truncated_flag: bool = False
            for action in solution:
                _, _, terminated, truncated_step, info = env.step(action)
                frames.append(env._render_rgb().copy())
                final_info = info
                truncated_flag = truncated_step
                if terminated or truncated_step:
                    break

            if final_info is not None:
                solved_flag: bool = bool(final_info.get("solved", False))
                deadlocked_flag: bool = bool(final_info.get("deadlocked", False))
            else:
                solved_flag = bool(env._is_solved())
                action_mask: np.ndarray = env._compute_action_mask()
                deadlocked_flag = bool(env._is_deadlocked(action_mask, solved_flag))

            truncated_flag = truncated_flag or (env._steps >= env.max_steps and not solved_flag and not deadlocked_flag)
            status_lines: list[str] = build_status_lines(env, solved_flag, deadlocked_flag, truncated_flag)
            frames[-1] = draw_status_overlay(frames[-1], status_lines)

            # Add a pause at the final frame by repeating it for `fps * final_pause_seconds` frames
            pause_multiplier: int = 3  # increase to make the pause more apparent
            pause_frames: int = max(1, int(max(0.0, final_pause_seconds) * max(1, int(fps)) * pause_multiplier))
            final_frame: NDArray[np.uint8] = frames[-1]
            # Some encoders/viewers aggressively collapse identical frames (GIF optimizations / Pillow deduplication).
            # To ensure the final frame pause remains visible we create imperceptible differences for each copy.
            # We alter a single pixel's alpha-equivalent channel by toggling a value within the valid range while
            # keeping the visual change negligible.
            frames_extended: list[NDArray[np.uint8]] = list(frames)
            h, w = final_frame.shape[:2]
            for i in range(pause_frames):
                copy_frame: NDArray[np.uint8] = final_frame.copy()
                # choose a pixel near the corner to minimize visible impact
                px_y: int = max(0, h - 1 - (i % max(1, h)))
                px_x: int = max(0, w - 1 - ((i // max(1, h)) % max(1, w)))
                # apply a tiny change that stays within [0,255] toggle the blue channel by +1/-1 depending on i parity
                delta: int = 1 if (i % 2 == 0) else -1
                old_val = int(copy_frame[px_y, px_x, 2])
                new_val: int = old_val + delta
                new_val = int(np.clip(new_val, 0, 255))
                copy_frame[px_y, px_x, 2] = np.array(new_val, dtype=np.uint8)
                frames_extended.append(copy_frame)

            file_name: str = f"{label}_mode-{mode}.gif"
            target: Path = output_dir / file_name
            iio3.imwrite(target, frames_extended, extension=".gif", duration=1.0 / max(fps, 1), loop=0)
            env.close()
            print(f"[INFO] Saved {target}")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Parse command-line arguments and generate GIFs for all configured NumberLink generator/variant combinations.

    :param argv: Optional list of argument strings to parse (defaults to ``sys.argv`` when ``None``).
    :type argv: list[str] | None
    :returns: None
    :rtype: None
    """
    parser: ArgumentParser = ArgumentParser(
        description="Generate NumberLink solution GIFs for configuration combinations."
    )
    parser.add_argument("--out", type=Path, default=Path("gifs"), help="Output directory for generated GIF files.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for level generation. If omitted generation is non-deterministic.",
    )
    parser.add_argument("--fps", type=int, default=4, help="Playback speed for the exported GIFs.")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Board size (width=height). If omitted, a sensible default is used based on variant.",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=None,
        help="Number of color pairs to generate. If omitted, a sensible default is used based on variant.",
    )
    parser.add_argument(
        "--final-pause-seconds",
        type=float,
        default=2.0,
        help="Seconds to pause on the final frame before the GIF loops.",
    )
    args: Namespace = parser.parse_args(argv)

    generate_gifs(
        args.out,
        seed=args.seed,
        fps=args.fps,
        size=args.size,
        colors=args.colors,
        final_pause_seconds=args.final_pause_seconds,
    )


if __name__ == "__main__":
    main()
