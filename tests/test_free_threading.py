from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import sysconfig
from typing import TYPE_CHECKING

from gymnasium.envs.registration import registry
import numpy as np
import pytest

from numberlink import GeneratorConfig, NumberLinkRGBEnv, NumberLinkRGBVectorEnv, VariantConfig
from numberlink.registration import register_numberlink_v0

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

FREE_THREADED: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
"""Whether the running interpreter was built without the global interpreter lock."""

gil_enabled: Callable[[], bool] | None = getattr(sys, "_is_gil_enabled", None)
"""Runtime probe for the global interpreter lock, available from Python 3.13 onward."""

requires_free_threading = pytest.mark.skipif(
    not FREE_THREADED, reason="requires a free-threaded (Py_GIL_DISABLED) interpreter"
)


def _drive_single_env(seed: int) -> tuple[int, bool]:
    """Run one single environment through several episodes and report the work it completed.

    A built-in level is used so the thread spends its time stepping the environment rather than generating a board.

    Args:
        seed: Seed used for the environment resets.

    Returns:
        Number of applied steps and whether every observation had the expected dtype and shape.
    """
    env = NumberLinkRGBEnv(render_mode="rgb_array", level_id="builtin_6x6_rw_5c")
    steps: int = 0
    shapes_ok: bool = True
    try:
        for episode in range(4):
            obs, _info = env.reset(seed=seed * 100 + episode)
            shapes_ok = shapes_ok and obs.dtype == np.uint8 and obs.ndim == 3 and obs.shape[-1] == 3
            mask: NDArray[np.uint8] = env.compute_action_mask()
            for _ in range(25):
                valid = np.flatnonzero(mask)
                if valid.size == 0:
                    break
                obs, _reward, terminated, truncated, _info = env.step(int(valid[0]))
                shapes_ok = shapes_ok and obs.dtype == np.uint8 and obs.ndim == 3
                steps += 1
                mask = env.compute_action_mask()
                if terminated or truncated:
                    break
    finally:
        env.close()
    return steps, shapes_ok


def _drive_vector_env(seed: int) -> tuple[int, bool]:
    """Run one vector environment through a batch of steps and report the work it completed.

    Args:
        seed: Seed used for the environment reset.

    Returns:
        Number of applied batch steps and whether every observation had the expected dtype and shape.
    """
    num_envs: int = 3
    env = NumberLinkRGBVectorEnv(num_envs=num_envs, render_mode="rgb_array", level_id="builtin_6x6_rw_5c")
    steps: int = 0
    shapes_ok: bool = True
    try:
        obs, _info = env.reset(seed=seed)
        shapes_ok = obs.dtype == np.uint8 and obs.shape[0] == num_envs
        for _ in range(20):
            obs, _rewards, _terminated, _truncated, _infos = env.step(np.zeros(num_envs, dtype=np.int64))
            shapes_ok = shapes_ok and obs.dtype == np.uint8 and obs.shape[0] == num_envs
            steps += 1
    finally:
        env.close()
    return steps, shapes_ok


@requires_free_threading
def test_gil_is_disabled_at_runtime() -> None:
    """Confirm the interpreter really runs without the global interpreter lock."""
    assert gil_enabled is not None
    assert gil_enabled() is False


@pytest.mark.parametrize("worker", [_drive_single_env, _drive_vector_env])
def test_environments_run_concurrently(worker: Callable[[int], tuple[int, bool]]) -> None:
    """Drive one environment per thread and confirm every thread completed its work correctly."""
    num_workers: int = 8
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(worker, range(1, num_workers + 1)))

    assert len(results) == num_workers
    for steps, shapes_ok in results:
        assert steps > 0
        assert shapes_ok


def _generate_level(seed: int) -> int:
    """Generate a board on a worker thread and report how many colors it contains.

    Args:
        seed: Seed handed to the generator.

    Returns:
        Number of colors on the generated board.
    """
    env = NumberLinkRGBEnv(
        render_mode="rgb_array",
        generator=GeneratorConfig(mode="random_walk", width=5, height=5, colors=3, seed=seed),
        variant=VariantConfig(must_fill=False),
    )
    try:
        env.reset(seed=seed)
        return int(env.num_colors)
    finally:
        env.close()


def test_level_generation_runs_concurrently() -> None:
    """Generate one board per thread and confirm every thread produced a usable board."""
    num_workers: int = 8
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        colors = list(pool.map(_generate_level, range(1, num_workers + 1)))

    assert len(colors) == num_workers
    assert all(count > 0 for count in colors)


def test_registration_is_thread_safe() -> None:
    """Register the same ids from many threads and confirm each id ends up registered exactly once."""
    env_ids: list[str] = [f"ThreadSafeNumberLink{index}-v0" for index in range(4)]
    for env_id in env_ids:
        registry.pop(env_id, None)

    def register_repeatedly(worker_index: int) -> None:
        for repeat in range(50):
            register_numberlink_v0(env_ids[(worker_index + repeat) % len(env_ids)])

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(register_repeatedly, range(16)))

        for env_id in env_ids:
            spec = registry.get(env_id)
            assert spec is not None
            assert spec.entry_point == "numberlink:NumberLinkRGBEnv"
            assert spec.vector_entry_point == "numberlink:NumberLinkRGBVectorEnv"
    finally:
        for env_id in env_ids:
            registry.pop(env_id, None)
