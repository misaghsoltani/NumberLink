"""Registration helper for the NumberLink environment.

Provides a convenience function to register the environment id ``NumberLinkRGB-v0`` with Gymnasium.
The ``numberlink`` package automatically calls :func:`register_numberlink_v0` during module initialization,
so simply importing the package is sufficient to register the environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium.envs.registration import register, registry

from .env import NumberLinkRGBEnv
from .vector_env import NumberLinkRGBVectorEnv

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from gymnasium.envs.registration import EnvSpec

    from .config import GeneratorConfig, RenderConfig, RewardConfig, VariantConfig
    from .types import Coord, RenderMode, RGBInt


def register_numberlink_v0(env_id: str = "NumberLinkRGB-v0") -> None:
    """Register the environment id with Gymnasium's registry.

    This function registers the NumberLink environment so it can be instantiated using ``gymnasium.make``.
    The registration is idempotent and safe to call multiple times. If the environment has already been
    registered, the function returns immediately without raising an error or duplicating the registration.

    The function registers both a standard environment entry point and a vectorized entry point that can
    be used with ``gymnasium.make_vec`` for parallel environment execution.

    .. note::
        The ``numberlink`` package automatically calls this function during module initialization, so
        simply importing the package (``import numberlink``) is sufficient to register the environment.
        Explicit calls to this function are only needed if you want to register with a custom ``env_id``
        or in rare cases where you need to ensure registration without importing the package.

    :param env_id: Environment identifier to register. Defaults to ``"NumberLinkRGB-v0"``.
    :type env_id: str, optional
    :raises ValueError: If the provided ``env_id`` is empty or improperly formatted according to Gymnasium
        conventions. The id should follow the pattern ``[namespace/](env_name)[-v(version)]``.

    Example usage::

        import gymnasium as gym
        import numberlink  # Automatically registers NumberLinkRGB-v0

        # Create a single environment
        env = gym.make("NumberLinkRGB-v0")

        # Or create a vectorized environment using gymnasium.make_vec
        vec_env = gym.make_vec("NumberLinkRGB-v0", num_envs=3)

    .. note::
        This function follows Gymnasium>=1.0 conventions and does not use deprecated parameters such as
        ``autoreset`` or ``apply_api_compatibility`` that were removed in Gymnasium 1.0.

    .. seealso::
        :func:`gymnasium.register`
            Core Gymnasium registration function used internally.

        :func:`gymnasium.make`
            Function to instantiate environments by id after registration.

        :func:`gymnasium.make_vec`
            Function to create vectorized environments using the registered vector entry point.
    """
    if not env_id:
        raise ValueError(f"env_id must be a non-empty string, got: {env_id!r}")

    # Check if already registered by querying the registry
    existing_spec: EnvSpec | None = registry.get(env_id)
    if existing_spec is not None:
        return

    register(
        id=env_id,
        entry_point="numberlink:NumberLinkRGBEnv",
        vector_entry_point="numberlink:NumberLinkRGBVectorEnv",
        kwargs={},
        max_episode_steps=None,
        disable_env_checker=False,
        order_enforce=True,
        nondeterministic=False,
    )


def env_creator(
    grid: Sequence[str] | None = None,
    *,
    render_mode: RenderMode | None = None,
    level_id: str | None = None,
    variant: VariantConfig | None = None,
    bridges: Iterable[Coord] | None = None,
    generator: GeneratorConfig | None = None,
    reward_config: RewardConfig | None = None,
    render_config: RenderConfig | None = None,
    step_limit: int | None = None,
    palette: dict[str, RGBInt] | None = None,
    solution: list[list[Coord]] | None = None,
) -> NumberLinkRGBEnv:
    """Entry-point factory for Gymnasium. Returns a NumberLinkRGBEnv instance.

    This function is suitable to reference from packaging entry points (``gymnasium.envs``) so Gymnasium can discover
    environments automatically when the package is installed.
    """
    return NumberLinkRGBEnv(
        grid=grid,
        render_mode=render_mode,
        level_id=level_id,
        variant=variant,
        bridges=bridges,
        generator=generator,
        reward_config=reward_config,
        render_config=render_config,
        step_limit=step_limit,
        palette=palette,
        solution=solution,
    )


def vector_env_creator(
    num_envs: int = 1,
    grid: Sequence[str] | None = None,
    *,
    render_mode: RenderMode | None = None,
    level_id: str | None = None,
    variant: VariantConfig | None = None,
    bridges: Iterable[Coord] | None = None,
    generator: GeneratorConfig | None = None,
    reward_config: RewardConfig | None = None,
    render_config: RenderConfig | None = None,
    step_limit: int | None = None,
    palette: dict[str, RGBInt] | None = None,
    solution: list[list[Coord]] | None = None,
) -> NumberLinkRGBVectorEnv:
    """Create a vectorized NumberLink environment.

    The function mirrors the standard constructor signature used by Gymnasium when building vector environments.
    """
    return NumberLinkRGBVectorEnv(
        num_envs=num_envs,
        grid=grid,
        render_mode=render_mode,
        level_id=level_id,
        variant=variant,
        bridges=bridges,
        generator=generator,
        reward_config=reward_config,
        render_config=render_config,
        step_limit=step_limit,
        palette=palette,
        solution=solution,
    )
