"""NumberLink Environment Package for Reinforcement Learning, built on Gymnasium.

The package exposes configuration dataclasses, environment classes, and the ``LEVELS`` mapping of built-in levels.
Import these symbols for your use or to register the environment using :func:`register_numberlink_v0`.

By default the package does not automatically register Gymnasium ids on import. Library consumers should prefer
explicit registration via ``numberlink.register_numberlink_v0()``. When installed from PyPI the package also
exposes packaging entry-points (``project.entry-points."gymnasium.envs"``) so Gymnasium can auto-discover and
load the environment without an explicit registration call. See the project ``pyproject.toml`` for the
``gymnasium.envs`` entry-point names.
"""

from __future__ import annotations

from .config import GeneratorConfig, RenderConfig, RewardConfig, VariantConfig
from .env import NumberLinkRGBEnv
from .levels import LEVELS
from .notebook_viewer import NumberLinkNotebookViewer
from .registration import register_numberlink_v0
from .vector_env import NumberLinkRGBVectorEnv
from .viewer import NumberLinkViewer

__version__ = "0.1.2"
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
