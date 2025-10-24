"""NumberLink Environment Package for Reinforcement Learning, built on Gymnasium.

The package exposes configuration dataclasses, environment classes, and the ``LEVELS`` mapping of built-in levels.
Import these symbols for your use or to register the environment using :func:`register_numberlink_v0`.

The package automatically registers the ``NumberLinkRGB-v0`` environment id with Gymnasium on import by calling
:func:`register_numberlink_v0` at module initialization. This means ``import numberlink`` is sufficient to make
the environment available for ``gymnasium.make("NumberLinkRGB-v0")``.

When installed from PyPI, the package also exposes packaging entry-points (``project.entry-points."gymnasium.envs"``)
so Gymnasium can discover and instantiate the environment through its entry-point system. See the project
``pyproject.toml`` for the ``gymnasium.envs`` entry-point definitions.
"""

from __future__ import annotations

from .config import GeneratorConfig, RenderConfig, RewardConfig, VariantConfig
from .env import NumberLinkRGBEnv
from .levels import LEVELS
from .notebook_viewer import NumberLinkNotebookViewer
from .registration import register_numberlink_v0
from .vector_env import NumberLinkRGBVectorEnv
from .viewer import NumberLinkViewer

__version__ = "0.1.5"
__author__ = "Misagh Soltani"

register_numberlink_v0()

__all__: list[str] = [
    "__version__",
    "__author__",
    "NumberLinkRGBEnv",
    "register_numberlink_v0",
    "GeneratorConfig",
    "VariantConfig",
    "RewardConfig",
    "RenderConfig",
    "LEVELS",
    "NumberLinkViewer",
    "NumberLinkNotebookViewer",
    "NumberLinkRGBVectorEnv",
]
