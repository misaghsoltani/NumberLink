from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from numberlink import GeneratorConfig, NumberLinkViewer, RenderConfig, VariantConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray


def main() -> None:
    """Run the human-play viewer for NumberLinkRGB-v0."""
    # Create an environment with human render mode and the built-in viewer
    env: gym.Env[NDArray[np.uint8], np.int64] = gym.make(
        "NumberLinkRGB-v0",
        render_mode="human",
        generator=GeneratorConfig(mode="hamiltonian", colors=7, width=8, height=8, min_path_length=3),
        variant=VariantConfig(must_fill=True, allow_diagonal=False, cell_switching_mode=False, bridges_enabled=False),
        render_config=RenderConfig(gridline_color=(60, 60, 60), gridline_thickness=1, show_endpoint_numbers=True),
        # generator=None,  # use a built-in level
        # level_id="builtin_7x7_ham_6c",
    )
    env.reset(seed=2)

    viewer = NumberLinkViewer(env, cell_size=64)
    viewer.loop()


if __name__ == "__main__":
    main()
