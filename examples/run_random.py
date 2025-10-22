from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

import numberlink

if TYPE_CHECKING:
    from typing import SupportsFloat

    from numpy.typing import NDArray


numberlink.register_numberlink_v0()

# Single environment example using Gymnasium
env: gym.Env[NDArray[np.uint8], np.int64] = gym.make(
    "NumberLinkRGB-v0",
    render_mode="ansi",
    generator=numberlink.GeneratorConfig(width=10, height=10, colors=12),
    variant=numberlink.VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False),
)
obs: NDArray[np.uint8]
info: dict[str, NDArray[np.uint8] | float | int | bool]
obs, info = env.reset()
print("obs shape:", obs.shape)
print("Render:\n", env.render())

terminated: bool
truncated: bool
terminated = truncated = False
ret = 0.0
while not (terminated or truncated):
    mask: NDArray[np.uint8] | float | int | bool = info["action_mask"]
    action: np.int64 = np.random.choice(np.flatnonzero(mask))
    reward: SupportsFloat
    obs, reward, terminated, truncated, info = env.step(action)
    ret += float(reward)

print("episode return:", ret)

# Vectorized creation using Gymnasium
vec_env = gym.make_vec(
    "NumberLinkRGB-v0",
    num_envs=4,
    render_mode="rgb_array",
    generator=numberlink.GeneratorConfig(width=8, height=8, colors=6),
)
obs, infos = vec_env.reset()
