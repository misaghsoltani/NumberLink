Usage
=====

This guide walks through common workflows with the NumberLink environments, including the single RGB environment, the
vectorized batch environment, and the pygame viewer. The examples assume the package is installed and importable. Refer
to :doc:`installation` for environment setup instructions.

.. contents::
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Single RGB Environment
----------------------

The :class:`numberlink.env.NumberLinkRGBEnv` class implements the `Gymnasium <https://gymnasium.farama.org/>`_ interface and yields RGB observations. Each
action encodes ``(color, head, direction)``, where every color supplies two heads representing its endpoints. Paths can
be extended from either endpoint at any time, and the viewer updates focus automatically when you click on a cell.

.. code-block:: python

   import gymnasium as gym

   # Gymnasium uses the package entry points to discover NumberLinkRGB-v0
   env = gym.make("NumberLinkRGB-v0", render_mode="rgb_array")

   import numpy as np
   observation, info = env.reset(seed=123)
   action_mask = info["action_mask"].astype(np.int8)

   terminated = False
   truncated = False
   while not (terminated or truncated):
      action = env.action_space.sample(mask=action_mask)
      observation, reward, terminated, truncated, info = env.step(action)
      action_mask = info["action_mask"]

   env.close()

When developing from source or within a notebook, call :func:`numberlink.registration.register_numberlink_v0` before
creating the environment so the id is present in the registry for the current process. The info dictionary includes the
binary action mask, the step counter, connectivity status per color, the level identifier (when available), and two
flags named ``solved`` and ``deadlocked``.

Customize gameplay and rendering by passing structured configuration objects:

.. code-block:: python

   import gymnasium as gym
   from numberlink import GeneratorConfig, RenderConfig, VariantConfig

   env = gym.make(
      "NumberLinkRGB-v0",
      render_mode="rgb_array",
      generator=GeneratorConfig(width=8, height=8, colors=5),
      variant=VariantConfig(must_fill=True, allow_diagonal=False, bridges_enabled=False),
      render_config=RenderConfig(cell_size=48),
   )

   observation, info = env.reset(seed=7)
   env.close()

See :class:`numberlink.config.GeneratorConfig`, :class:`numberlink.config.VariantConfig`, and
:class:`numberlink.config.RenderConfig` for the available fields. :mod:`numberlink.level_setup` documents helper
functions that assemble these objects from level templates.

Vector Environment
------------------

The :class:`numberlink.vector_env.NumberLinkRGBVectorEnv` class runs multiple puzzles in parallel. It shares all
configuration objects with the single environment and exposes batched observations and rewards.

.. code-block:: python

   import gymnasium as gym
   from numberlink import GeneratorConfig

   vec_env = gym.make_vec(
      "NumberLinkRGB-v0",
      num_envs=8,
      render_mode="rgb_array",
      generator=GeneratorConfig(width=6, height=6, colors=4),
   )

   observations, infos = vec_env.reset(seed=7)
   import numpy as np
   actions = [vec_env.single_action_space.sample(mask=mask.astype(np.int8)) for mask in infos["action_mask"]]
   observations, rewards, terminated, truncated, infos = vec_env.step(actions)
   vec_env.close()

When a batch element reaches a terminal state, the vector environment auto-resets it on the next step. The returned info
dict mirrors the single-environment keys but stores arrays of shape ``(num_envs, ...)``.

Viewer and Human Mode
---------------------

The :mod:`numberlink.viewer` module provides a `pygame <https://www.pygame.org/>`_ viewer that mirrors the human render mode. It supports both mouse
and keyboard control. Clicking any endpoint or occupied cell updates the focus to that color and head, so every path can
be extended from either endpoint without using keyboard shortcuts.

.. code-block:: python

   import gymnasium as gym
   import numberlink
   from numberlink.viewer import NumberLinkViewer

   numberlink.register_numberlink_v0()
   env = gym.make("NumberLinkRGB-v0", render_mode="human")
   viewer = NumberLinkViewer(env, cell_size=64)
   viewer.loop()

Default controls include arrow keys (and ``Q``/``E``/``Z``/``C`` for diagonals when enabled), brackets to pin a specific
head, :kbd:`Tab` to cycle colors, and :kbd:`Space` to backtrack the active head by one cell. In cell switching mode the
cursor follows mouse clicks, and painting obeys the active color and configuration.

Notebook environments (Jupyter, JupyterLab, Google Colab) can render the same controls inline when the optional
extra ``numberlink[notebook]`` is installed. Either instantiate
:class:`numberlink.notebook_viewer.NumberLinkNotebookViewer` directly, or call
:meth:`numberlink.viewer.NumberLinkViewer.loop` and the backend will automatically switch to the widget-based viewer.

.. code-block:: python

   env = gym.make(
      "NumberLinkRGB-v0",
      render_mode="human",
      generator=GeneratorConfig(
         mode="hamiltonian",
         colors=7,
         width=8,
         height=8,
         must_fill=True,
         min_path_length=3,
      ),
      variant=VariantConfig(
         allow_diagonal=False, cell_switching_mode=False, bridges_enabled=False
      ),
      render_config=RenderConfig(
         gridline_color=(60, 60, 60),
         gridline_thickness=1,
         show_endpoint_numbers=True,
         render_height=400,
         render_width=400,
      ),
   )
   env.reset(seed=2)

   viewer = NumberLinkViewer(env, cell_size=64)
   viewer.loop()

If the extras are missing, the viewer emits a short installation hint instead of trying to open a pygame window in the
notebook runtime.

.. figure:: ../output/vector_scenarios.gif
   :alt: Batched environments advancing in parallel.
   :width: 70%

   Vector environments make it easy to step many puzzles at once for data collection or evaluation.

Command Line Interface
----------------------

The project ships a command-line interface so you can launch the viewer or inspect boards without writing code. Run
``python -m numberlink --help`` to see all subcommands.

``viewer`` launches the interactive pygame window. You can load built-in levels or point the CLI to a custom grid file.

.. code-block:: bash

   python -m numberlink viewer --level-id 6x6_rgb_2 --cell-size 72 --apply-solution

``board`` prints a text rendering of the puzzle and optionally applies the stored solution.

.. code-block:: bash

   python -m numberlink board --level-id builtin_7x7_ham_6c --apply-solution

``levels`` lists every bundled level id, and ``register`` registers the Gymnasium id ``NumberLinkRGB-v0`` for external
use. All commands share the same variant flags (for example ``--allow-diagonal`` or ``--bridges-enabled``) so you can
inspect the same configuration interactively and in text mode.

Configuration Overview
----------------------

NumberLink uses structured dataclasses to describe gameplay and rendering:

- :class:`numberlink.config.VariantConfig` toggles bridges, diagonal movement, cell switching mode, and full coverage.
- :class:`numberlink.config.RewardConfig` defines the step, invalid, connect, and win rewards.
- :class:`numberlink.config.RenderConfig` controls resolution, gridlines, palette adjustments, and endpoint numbering.
  Endpoint numbers are centered both vertically and horizontally and sized to roughly one third of a cell when
  auto-scaled.

Customize these objects directly or through helper functions in :mod:`numberlink.level_setup` before constructing an
environment.

Artifacts and Further Reading
-----------------------------

- Sample level definitions and generator presets live in :mod:`numberlink.level_setup` and :mod:`numberlink.generator`.
- Example scripts under ``examples/`` demonstrate solving, rendering, and evaluation patterns.
- The ``tests/`` directory includes unit tests for cell switching, vectorized execution, rendering, and configuration
  utilities.
