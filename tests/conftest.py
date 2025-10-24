from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from numberlink import GeneratorConfig, RenderConfig, VariantConfig


def pytest_configure(config: pytest.Config) -> None:
    """Ensure common output directory exists for generated GIFs."""
    Path("output").mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def output_dir() -> Path:
    """Return the output directory path, creating it if missing."""
    d = Path("output")
    d.mkdir(exist_ok=True)
    return d


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Save a list of RGB frames into a GIF at the provided path."""
    if not frames:
        return
    pil_frames: list[Image.Image] = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path, save_all=True, append_images=pil_frames[1:], duration=max(1, int(1000 / max(1, fps))), loop=0
    )


def base_render_config(
    h: int, w: int, ppc: int = 12, show_numbers: bool = False, font_max: int | None = None
) -> RenderConfig:
    """Create a RenderConfig scaled to the grid size for crisp visuals in tests.

    The optional ``font_max`` parameter allows tests to exercise both bounded and unbounded
    font sizing (``None`` means automatic sizing with no explicit upper bound).
    """
    return RenderConfig(
        render_height=h * ppc,
        render_width=w * ppc,
        endpoint_border_thickness=1,
        endpoint_border_color=(255, 255, 255),
        gridline_color=(60, 60, 60),
        gridline_thickness=1,
        show_endpoint_numbers=show_numbers,
        number_font_min_scale=1,
        number_font_max_scale=font_max,
    )


def standard_scenarios() -> list[tuple[str, GeneratorConfig, VariantConfig, bool]]:
    """Return a representative set of scenarios covering all feature flags."""
    # name, generator, variant, show_numbers
    return [
        (
            "standard_must_fill",
            GeneratorConfig(width=6, height=6, colors=4, seed=42),
            VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False),
            False,
        ),
        (
            "partial_fill",
            GeneratorConfig(width=6, height=6, colors=3, seed=123),
            VariantConfig(must_fill=False, allow_diagonal=False, bridges_enabled=False),
            False,
        ),
        (
            "diagonal",
            GeneratorConfig(width=6, height=6, colors=3, seed=456),
            VariantConfig(must_fill=True, allow_diagonal=True, bridges_enabled=False),
            False,
        ),
        (
            "bridges",
            GeneratorConfig(width=7, height=7, colors=4, bridges_probability=0.2, seed=789),
            VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=True),
            False,
        ),
        (
            "numbers",
            GeneratorConfig(width=6, height=6, colors=4, seed=321),
            VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False),
            True,
        ),
    ]
