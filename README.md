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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_interactive.ipynb)

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

- Home page & documentation: [NumberLink](https://misaghsoltani.github.io/NumberLink/)
- Quick start: [Quick Start](#quick-start)
- Google Colab: [Open in Colab](https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_interactive.ipynb)
- Installation guide: [Installation - documentation](https://misaghsoltani.github.io/NumberLink/installation.html)
- CLI reference: [CLI - documentation](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.cli.html)
- Python API: [API reference - documentation](https://misaghsoltani.github.io/NumberLink/apidocs/index.html)
- Citing this project: [Cite this project](#cite-this-work)
- Contact: [Contact](#contact)

### Demo

| ![Must fill](https://raw.githubusercontent.com/misaghsoltani/NumberLink/master/docs/_static/gifs/quickstart_must_fill.gif) | ![Cell switching](https://raw.githubusercontent.com/misaghsoltani/NumberLink/master/docs/_static/gifs/quickstart_cell_switching.gif) | ![Path mode](https://raw.githubusercontent.com/misaghsoltani/NumberLink/master/docs/_static/gifs/quickstart_path.gif) | ![Bridges and diagonal](https://raw.githubusercontent.com/misaghsoltani/NumberLink/master/docs/_static/gifs/quickstart_bridges_diagonal.gif) |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |

## Quick start

The [NumberLink documentation](https://misaghsoltani.github.io/NumberLink/) covers every workflow in detail. The
highlights below show the recommended [Gymnasium](https://gymnasium.farama.org/) >= 1.0 usage patterns. You can also try it out in the [Google Colab example](https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_interactive.ipynb).

### Install from PyPI

#### Using pip

```bash
pip install numberlink
```

#### Install with [uv](https://docs.astral.sh/uv/)

```bash
uv pip install numberlink
```

#### Enable notebook integration

Install the optional notebook dependencies to enable inline controls in Jupyter and Google Colab:

```bash
pip install "numberlink[notebook]"
```

With `uv`:

```bash
uv pip install "numberlink[notebook]"
```

See the [installation guide](https://misaghsoltani.github.io/NumberLink/installation.html) for Pixi, Conda, and source build instructions.

### Create a single environment

```python
import gymnasium as gym
import numpy as np

import numberlink  # Importing the package automatically registers the environments

env = gym.make("NumberLinkRGB-v0", render_mode="rgb_array")

observation, info = env.reset(seed=42)
action_mask = info["action_mask"]

terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample(mask=action_mask)
    observation, reward, terminated, truncated, info = env.step(action)
    action_mask = info["action_mask"].astype(np.int8)

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
from numberlink import GeneratorConfig  # Importing from numberlink automatically registers
import numpy as np

vec_env = gym.make_vec(
    "NumberLinkRGB-v0",
    num_envs=4,
    render_mode="rgb_array",
    generator=GeneratorConfig(width=6, height=6, colors=4),
)

observations, infos = vec_env.reset(seed=0)
actions = [vec_env.single_action_space.sample(mask=mask.astype(np.int8)) for mask in infos["action_mask"]]
observations, rewards, terminated, truncated, infos = vec_env.step(actions)
vec_env.close()
```

Gymnasium auto resets terminated slots when the vector environment is configured with the default autoreset mode. See
the [vector API section of the docs](https://misaghsoltani.github.io/NumberLink/usage.html#vector-environment) for
batched workflows.

### Human render mode

```python
import gymnasium as gym
import numberlink  # Auto-registration on import
from numberlink.viewer import NumberLinkViewer

env = gym.make("NumberLinkRGB-v0", render_mode="human")
viewer = NumberLinkViewer(env)
viewer.loop()
```

The pygame viewer mirrors the CLI command shown in
[examples/run_human.py](https://github.com/misaghsoltani/NumberLink/blob/main/examples/run_human.py) and is documented
at [viewer API](https://misaghsoltani.github.io/NumberLink/apidocs/numberlink/numberlink.viewer.html).

### Notebook viewer

The package ships an inline viewer that mirrors the pygame controls when the optional notebook extras are installed.

```python
env = gym.make(
    "NumberLinkRGB-v0",
    render_mode="human",
    generator=GeneratorConfig(mode="hamiltonian", colors=7, width=8, height=8, min_path_length=3),
    variant=VariantConfig(must_fill=True, allow_diagonal=False, cell_switching_mode=False, bridges_enabled=False),
    render_config=RenderConfig(gridline_color=(60, 60, 60),gridline_thickness=1,show_endpoint_numbers=True,render_height=400,render_width=400),
)
env.reset()

viewer = NumberLinkViewer(env, cell_size=64)
viewer.loop()
```

`NumberLinkViewer.loop()` auto-detects notebook runtimes. When the optional dependencies are available it displays the
inline widgets, otherwise it shows an installation hint so the classic pygame window is not opened in headless
contexts.

## Auto-registration

The package automatically registers the `NumberLinkRGB-v0` environment with Gymnasium when you import it:

```python
import numberlink
```

Additionally, when installed from PyPI, the package provides Gymnasium entry points that allow environment discovery
without explicit imports. This dual mechanism ensures the environments are available whether you import the package
directly or rely on Gymnasium's entry-point system.

See the [documentation](https://misaghsoltani.github.io/NumberLink/) for more details.

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
