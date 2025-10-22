# NumberLink Environment for Gymnasium

[![image](https://img.shields.io/pypi/v/numberlink.svg)](https://pypi.python.org/pypi/numberlink)
[![image](https://img.shields.io/pypi/l/numberlink.svg)](https://github.com/misaghsoltani/NumberLink/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/pyversions/numberlink.svg)](https://pypi.python.org/pypi/numberlink)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&label=package%20manager)](https://pixi.sh)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
![Static Badge](https://img.shields.io/badge/statically%20typed-mypy-039dfc)
[![Build & Publish](https://github.com/misaghsoltani/NumberLink/actions/workflows/publish_to_pypi.yml/badge.svg)](https://github.com/misaghsoltani/NumberLink/actions/workflows/publish_to_pypi.yml)
[![Deploy Documentation](https://github.com/misaghsoltani/NumberLink/actions/workflows/docs.yml/badge.svg)](https://github.com/misaghsoltani/NumberLink/actions/workflows/docs.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_quickstart.ipynb)

<br/>

<p align="center">
  <img alt="NumberLink Logo" src="https://raw.githubusercontent.com/misaghsoltani/NumberLink/master/docs/_static/numberlink-logo.svg" />
</p>

<br/>

A Gymnasium environment for the NumberLink puzzle game.

## Gameplay Rules

NumberLink connects matching endpoints with non overlapping paths on a grid.

NumberLink boards follow these invariants:

- Every pair of endpoints must be connected by a single path. Endpoints are enumerated in `numberlink.level_setup.LevelTemplate` and copied into the environment state.
- Paths cannot branch or reuse grid cells. The environment enforces this through the action mask returned by `numberlink.env.NumberLinkRGBEnv.reset` and `numberlink.env.NumberLinkRGBEnv.step`.
- Unless the chosen variant disables the requirement, every cell must belong to a path. Toggle this rule with `numberlink.config.VariantConfig.must_fill`.
- Bridge cells yield independent vertical and horizontal lanes governed by `numberlink.config.VariantConfig.bridges_enabled`.
- Diagonal moves are allowed only when `numberlink.config.VariantConfig.allow_diagonal` is set. Cell switching is controlled by `numberlink.config.VariantConfig.cell_switching_mode`.

## Quick links

- Quick start: [Quick Start](#quick-start)
- Google Colab: [Open in Colab](https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_quickstart.ipynb)
- Installation guide: [Installation - documentation site](https://misaghsoltani.github.io/NumberLink/installation.html)
- CLI reference: [CLI - documentation site](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.cli.html)
- Python usage (API snippets): [API reference - documentation site](https://misaghsoltani.github.io/NumberLink/apidocs/index.html)
- Citing this project: [Cite this project](#cite-this-work)
- Contact: [Contact](#contact)

### Demo

| ![Must fill](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/docs/_static/gifs/quickstart_must_fill.gif) | ![Cell switching](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/docs/_static/gifs/quickstart_cell_switching.gif) | ![Path mode](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/docs/_static/gifs/quickstart_path.gif) | ![Bridges and diagonal](https://raw.githubusercontent.com/misaghsoltani/DeepCubeAI/master/docs/_static/gifs/quickstart_bridges_diagonal.gif) |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |

## Quick start

The [NumberLink documentation](https://misaghsoltani.github.io/NumberLink/) covers every workflow in detail. The
highlights below show the recommended [Gymnasium](https://gymnasium.farama.org/) >= 1.0 usage patterns. You can also try it out in the [Google Colab example](https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_quickstart.ipynb).

### Install from PyPI

#### Using pip

```bash
pip install numberlink
```

#### Install with [uv](https://docs.astral.sh/uv/)

```bash
uv pip install numberlink
```

See the [installation guide](https://misaghsoltani.github.io/NumberLink/installation.html) for Pixi, Conda, and source build instructions.

### Create a single environment

```python
import gymnasium as gym

# Gymnasium discovers NumberLinkRGB-v0 from the package entry points
env = gym.make("NumberLinkRGB-v0", render_mode="rgb_array")

observation, info = env.reset(seed=42)
action_mask = info["action_mask"]

terminated = False
truncated = False
while not (terminated or truncated):
    action = env.action_space.sample(mask=action_mask)
    observation, reward, terminated, truncated, info = env.step(action)
    action_mask = info["action_mask"]

env.close()
```

Configuration objects such as
[GeneratorConfig](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.config.html#numberlink.config.GeneratorConfig),
[VariantConfig](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.config.html#numberlink.config.VariantConfig),
and [RenderConfig](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.config.html#numberlink.config.RenderConfig)
customize generation, gameplay rules, and rendering. Examples live in the
[usage guide](https://misaghsoltani.github.io/NumberLink/usage.html) and the
[level setup module](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.level_setup.html).

### Run vectorized environments

```python
import gymnasium as gym
from numberlink import GeneratorConfig

vec_env = gym.make_vec(
    "NumberLinkRGB-v0",
    num_envs=4,
    render_mode="rgb_array",
    generator=GeneratorConfig(width=6, height=6, colors=4),
)

observations, infos = vec_env.reset(seed=0)
actions = [vec_env.single_action_space.sample(mask=mask) for mask in infos["action_mask"]]
observations, rewards, terminated, truncated, infos = vec_env.step(actions)
vec_env.close()
```

Gymnasium auto resets terminated slots when the vector environment is configured with the default autoreset mode. See
the [vector API section of the docs](https://misaghsoltani.github.io/NumberLink/usage.html#vector-environment) for
batched workflows.

### Human render mode

```python
import gymnasium as gym
import numberlink
from numberlink.viewer import NumberLinkViewer

numberlink.register_numberlink_v0()
env = gym.make("NumberLinkRGB-v0", render_mode="human")
viewer = NumberLinkViewer(env)
viewer.loop()
```

The pygame viewer mirrors the CLI command shown in
[examples/run_human.py](https://github.com/misaghsoltani/NumberLink/blob/main/examples/run_human.py) and is documented
at [viewer API](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.viewer.html).

## Auto-registration

Recommended usage is to install the package (for example via PyPI), and Gymnasium then discovers the environments via
the package's entry-points and you can call `gymnasium.make(...)` / `gymnasium.make_vec(...)` directly. If you need to
register the env id in-process (for development or interactive use), call:

```python
import numberlink
numberlink.register_numberlink_v0()
```

See the docs for details on packaging entry-points and the registration helper:
[https://misaghsoltani.github.io/NumberLink/usage.html](https://misaghsoltani.github.io/NumberLink/usage.html)
and [https://misaghsoltani.github.io/NumberLink/apidocs/index.html](https://misaghsoltani.github.io/NumberLink/apidocs/index.html).

## License

MIT License - see [LICENSE](https://github.com/misaghsoltani/NumberLink/blob/main/LICENSE).

## Cite this work

If you use NumberLink in your research, please cite:

<!-- CITATION-BIBTEX:START -->
```bibtex
@misc{numberlinkenv2025soltani,
  author = {Soltani, Misagh},
  title  = {NumberLink Puzzle},
  url    = {https://misaghsoltani.github.io/NumberLink},
  year   = {2025}
}
```
<!-- CITATION-BIBTEX:END -->

## Contact

If you have any questions or issues, please contact Misagh Soltani ([misaghsoltani@gmail.com](mailto:misaghsoltani@gmail.com))
