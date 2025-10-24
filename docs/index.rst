.. _index:

NumberLink Documentation
=================================

.. figure:: _static/numberlink-logo.svg
   :alt: NumberLink logo
   :width: 45%

   NumberLink connects matching endpoints with non overlapping paths on a grid.

:mod:`numberlink` provides a `Gymnasium <https://gymnasium.farama.org/>`_ RGB environment, a vectorized batch variant, and a pygame viewer described in :doc:`usage`. This environment was inspired by `Puzzle Baron's NumberLinks <https://numberlinks.puzzlebaron.com>`_.

Gameplay Rules
--------------

NumberLink boards follow these invariants:

- Every pair of endpoints must be connected by a single path. Endpoints are enumerated in :class:`numberlink.level_setup.LevelTemplate` and copied into the environment state.
- Paths cannot branch or reuse grid cells. The environment enforces this through the action mask returned by :meth:`numberlink.env.NumberLinkRGBEnv.reset` and :meth:`numberlink.env.NumberLinkRGBEnv.step`.
- Unless the chosen variant disables the requirement, every cell must belong to a path. Toggle this rule with :attr:`numberlink.config.VariantConfig.must_fill`.
- Bridge cells yield independent vertical and horizontal lanes governed by :attr:`numberlink.config.VariantConfig.bridges_enabled`.
- Diagonal moves are allowed only when :attr:`numberlink.config.VariantConfig.allow_diagonal` is set. Cell switching is controlled by :attr:`numberlink.config.VariantConfig.cell_switching_mode`.

.. image:: _static/gifs/quickstart_must_fill.gif
   :alt: Must fill variant example
   :width: 160px
   :class: quickstart-gif

.. image:: _static/gifs/quickstart_cell_switching.gif
   :alt: Cell switching variant example
   :width: 160px
   :class: quickstart-gif

.. image:: _static/gifs/quickstart_path.gif
   :alt: Default path building example
   :width: 160px
   :class: quickstart-gif

.. image:: _static/gifs/quickstart_bridges_diagonal.gif
   :alt: Bridges and diagonal variant example
   :width: 160px
   :class: quickstart-gif

Quick Install
-------------

Install the published package from `PyPI <https://pypi.org/project/numberlink/>`_:

.. code-block:: console

   pip install numberlink

For a reproducible workflow, `uv <https://docs.astral.sh/uv/>`_ can manage the virtual environment and dependencies:

.. code-block:: console

   uv pip install numberlink

See :doc:`installation` for Conda, Pixi, and source builds.

Quick Start
-----------

Explore the workflows below or launch the interactive `Google Colab example <https://colab.research.google.com/github/misaghsoltani/NumberLink/blob/main/notebooks/numberlink_interactive.ipynb>`_.

Setup Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   import numpy as np

   import numberlink  # Importing the package automatically registers NumberLinkRGB-v0

   env = gym.make("NumberLinkRGB-v0", render_mode="rgb_array")

   observation, info = env.reset(seed=42)
   action_mask = info["action_mask"].astype(np.int8)
   terminated = False
   truncated = False
   while not (terminated or truncated):
      action = env.action_space.sample(mask=action_mask)
      observation, reward, terminated, truncated, info = env.step(action)
      action_mask = info["action_mask"]
   env.close()

The ``import numberlink`` statement automatically registers the environment id with Gymnasium, making it immediately
available for use. When installed from PyPI, Gymnasium can also discover the environment through package entry points
without needing an explicit import.

Vectorized execution
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   import numpy as np

   import numberlink  # Auto-registration on import

   vec_env = gym.make_vec("NumberLinkRGB-v0", num_envs=4, render_mode="rgb_array")

   observations, infos = vec_env.reset(seed=7)
   actions = [vec_env.single_action_space.sample(mask=mask.astype(np.int8)) for mask in infos["action_mask"]]
   observations, rewards, terminated, truncated, infos = vec_env.step(actions)
   vec_env.close()

Single environments and vector environments share :class:`numberlink.config.GeneratorConfig`, :class:`numberlink.config.VariantConfig`, and :class:`numberlink.config.RenderConfig`. See :doc:`usage` for parameter tables and composition utilities in :mod:`numberlink.level_setup`.

Human mode viewer
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   import numberlink  # Auto-registration on import
   from numberlink.viewer import NumberLinkViewer

   env = gym.make("NumberLinkRGB-v0", render_mode="human")
   viewer = NumberLinkViewer(env)
   viewer.loop()

.. toctree::
   :maxdepth: 2

   installation
   usage
   apidocs/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
