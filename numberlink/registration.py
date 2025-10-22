"""Registration helper for the NumberLink environment.

Provides a convenience function to register the environment id ``NumberLinkRGB-v0`` with Gymnasium.
Call :func:`register_numberlink_v0` before creating environments with ``gymnasium.make``.

For more documentation see: https://misaghsoltani.github.io/NumberLink/ and the
Gymnasium docs at https://gymnasium.farama.org/.
"""

from __future__ import annotations

from typing import Any

from gymnasium.envs.registration import register

from .env import NumberLinkRGBEnv
from .vector_env import NumberLinkRGBVectorEnv


def register_numberlink_v0(env_id: str = "NumberLinkRGB-v0") -> None:
    """Register the environment id with Gymnasium's registry.

    :param env_id: Environment identifier to register. Defaults to ``"NumberLinkRGB-v0"``.
    :type env_id: str, optional

    Example usage::

        import numberlink as nle
        import gymnasium as gym

        # Register the env id, then create a single env
        nle.register_numberlink_v0()
        env = gym.make("NumberLinkRGB-v0")

        # Or create a vectorized environment using gymnasium.make_vec
        vec_env = gym.make_vec("NumberLinkRGB-v0", num_envs=3)
    """
    register(
        id=env_id,
        entry_point="numberlink:NumberLinkRGBEnv",
        vector_entry_point="numberlink:NumberLinkRGBVectorEnv",
        kwargs={},
        max_episode_steps=None,
    )


def env_creator(**kwargs: Any) -> NumberLinkRGBEnv:
    """Entry-point factory for Gymnasium. Returns a NumberLinkRGBEnv instance.

    This function is suitable to reference from packaging entry points (``gymnasium.envs``)
    so Gymnasium can discover environments automatically when the package is installed.
    """
    return NumberLinkRGBEnv(**kwargs)


def vector_env_creator(num_envs: int = 1, **kwargs: Any) -> NumberLinkRGBVectorEnv:
    """Create a vectorized NumberLink environment.

    The function mirrors the standard constructor signature used by Gymnasium
    when building vector environments.
    """
    return NumberLinkRGBVectorEnv(num_envs=num_envs, **kwargs)
