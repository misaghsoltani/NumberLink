Installation
============

NumberLink provides a Python package (``numberlink``) and CLI (``numberlink-cli``).

* **Supported Python:** 3.10+

1. Install ``numberlink`` Package from PyPI
-------------------------------------------

``numberlink`` is available on PyPI and supports **Python 3.10+**.

Option A - Install with `uv <https://docs.astral.sh/uv/>`_ (Recommended If Using the Package)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. **Install `uv`** from the official website: `Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_.
#. Create and activate a virtual environment:

   .. code-block:: bash

      # create a .venv in the current folder
      uv venv
      # macOS & Linux
      source .venv/bin/activate
      # Windows (PowerShell)
      .venv\Scripts\activate

   If you have multiple Python versions, ensure you use a supported one (3.10+), e.g.:

   .. code-block:: bash

      uv venv --python 3.14

#. Install the package (using `uv's pip interface <https://docs.astral.sh/uv/pip/>`_):

   .. code-block:: bash

      uv pip install numberlink

Option B - Install with `pip <https://pip.pypa.io/en/stable/>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Install pip from the official website: `Install pip <https://pip.pypa.io/en/stable/installation/>`_.
#. (Recommended) Create and activate a virtual environment.

   Create a ``.venv`` in the current folder:

   .. code-block:: bash

      python -m venv .venv

   Activate the virtual environment:

   .. code-block:: bash

      # macOS & Linux
      source .venv/bin/activate

      # Windows (PowerShell)
      .venv\Scripts\activate

   Install the package:

   .. code-block:: bash

      pip install numberlink

   See: `pip install command <https://pip.pypa.io/en/stable/cli/pip_install/>`_.

2. Source install with Pixi (Recommended when Installing from Source)
---------------------------------------------------------------------

`Pixi <https://pixi.sh/>`_ is a package management tool that provides fast, reproducible environments with support for Conda and PyPI dependencies. The ``pixi.toml`` (`pixi.toml on GitHub <https://github.com/misaghsoltani/NumberLink/blob/main/pixi.toml>`_) and ``pixi.lock`` (`pixi.lock on GitHub <https://github.com/misaghsoltani/NumberLink/blob/main/pixi.lock>`_) files define reproducible environments with exact dependency versions.

Installation steps
~~~~~~~~~~~~~~~~~~

#. **Install Pixi**: Follow the `official Pixi installation guide <https://pixi.sh/latest/installation/>`_.
#. **Clone repository**:

   .. code-block:: bash

      git clone https://github.com/misaghsoltani/NumberLink.git
      cd NumberLink

#. **Enter the default environment** (first run performs dependency resolution):

   .. code-block:: bash

      pixi shell          # or: pixi shell -e default
      # non-interactive solve only:
      pixi install -e default

#. **Verify installation**:

   .. code-block:: bash

      numberlink-cli --help

2.1 Available environments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pixi environments are defined in the ``[environments]`` section of ``pixi.toml`` (`pixi.toml on GitHub <https://github.com/misaghsoltani/NumberLink/blob/main/pixi.toml>`_). Each environment includes different feature sets for specific use cases:

+----------+----------------------------------------------------+
| Name     | Description                                        |
+==========+====================================================+
| default  | Core runtime dependencies                          |
+----------+----------------------------------------------------+
| dev      | Development tools: ruff, mypy, pyright, shellcheck |
+----------+----------------------------------------------------+
| build    | Build tools (hatch)                                |
+----------+----------------------------------------------------+
| all      | Complete development environment (dev, build)      |
+----------+----------------------------------------------------+
| glibc217 | All features with glibc 2.17 compatibility         |
+----------+----------------------------------------------------+
| doc      | Documentation build tools (Sphinx and helpers)     |
+----------+----------------------------------------------------+
| test     | Testing tools: pytest, pytest-cov, nose2, pillow   |
+----------+----------------------------------------------------+

**Activate an environment**:

.. code-block:: bash

   pixi shell -e dev
   pixi shell -e all

All environments share the same solve-group (``default``) for consistent dependency resolution. See `Pixi's environment documentation <https://pixi.sh/latest/features/environment/>`_ for more details.

2.2 Development tasks
~~~~~~~~~~~~~~~~~~~~~

The ``dev`` feature includes predefined `tasks <https://pixi.sh/latest/workspace/advanced_tasks/>`_ for code quality and type checking. Run these commands inside an environment that includes the ``dev`` feature:

.. code-block:: bash

   pixi run -e dev lint       # ruff check --fix
   pixi run -e dev ulint      # ruff check --fix --unsafe-fixes
   pixi run -e dev format     # ruff format
   pixi run -e dev fix        # ruff check --fix --unsafe-fixes followed by ruff format
   pixi run -e dev mypy       # mypy type check on 'numberlink/'
   pixi run -e dev pyright    # pyright type check on 'numberlink/'
   pixi run -e dev format-check  # ruff format --check
   pixi run -e dev typecheck     # pyright && mypy numberlink
   pixi run -e dev yamllint      # yamllint .
   pixi run -e dev shellcheck    # shellcheck on scripts (if present)
   pixi run -e dev check         # full checks (ruff, pyright, mypy, yamllint, shellcheck)
   pixi run -e dev update-citation  # update citation file (if `scripts/update_citation.sh` exists)

The ``doc`` feature contains documentation-related tasks (see ``pixi.toml``). Example commands for building the docs inside the ``doc`` environment:

.. code-block:: bash

   pixi run -e doc docs       # build docs
   pixi run -e doc docs-nitpick  # build docs with nitpicky warnings enabled
   pixi run -e doc docs-run   # build docs and serve locally (may try to bind port 8000)

2.3 Running the project
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pixi run numberlink-cli -h

2.4 Building distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the build environment for creating distribution packages:

.. code-block:: bash

   pixi shell -e build
   pixi run build   # hatch build -t wheel -t sdist

   # Or in a single command:
   pixi run -e build build

Alternatively, invoke hatch directly if available in your PATH:

.. code-block:: bash

   hatch build -t wheel -t sdist

Distribution artifacts will be created in the ``dist/`` directory.

3. PyPI (binary / sdist) install
--------------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install numberlink
   numberlink-cli --help

4. Source install with uv
-------------------------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager and project manager that can replace pip, virtualenv, and other tools. It provides fast dependency resolution and environment management.

Setup steps
~~~~~~~~~~~

#. **Install UV**: Follow the `official UV installation guide <https://docs.astral.sh/uv/getting-started/installation/>`_.
#. **Clone repository**:

   .. code-block:: bash

      git clone https://github.com/misaghsoltani/NumberLink.git
      cd NumberLink

#. **Install the project**:

   .. code-block:: bash

      # uv will automatically create a virtual environment and install dependencies
      uv sync

      # Activate the environment
      source .venv/bin/activate  # Linux/macOS
      # Or on Windows: .venv\Scripts\activate

#. **Verify installation**:

   .. code-block:: bash

      uv run numberlink-cli --help
      # or after activation (source .venv/bin/activate)
      numberlink-cli --help

See `uv's documentation <https://docs.astral.sh/uv/>`_ for more usage and features.

5. Conda
--------

Use the provided environment files:

.. code-block:: bash

   # Default
   conda env create -f environment.yml -n numberlink

   # Development (adds lint/type tools)
   conda env create -f environment_dev.yml -n numberlink_dev

This will install the required packages.

Activate the environment:

.. code-block:: bash

   conda activate numberlink   # or: conda activate numberlink_dev

Or you can install from source within a Conda environment:

.. code-block:: bash

   # Editable source install
   uv pip install -e . # Using uv
   # or
   pip install -e . # Using pip

Verify Installation
-------------------

**Check installation**:

For package:

.. code-block:: bash

   python -m pip show numberlink || python -c "import numberlink\nprint(numberlink.__version__)"

For CLI:

.. code-block:: bash

   numberlink-cli --help

**Quick run**:

.. code-block:: bash

   numberlink-cli viewer

Dependencies
------------

**Core Python dependencies** (see ``pixi.toml`` (`pixi.toml on GitHub <https://github.com/misaghsoltani/NumberLink/blob/main/pixi.toml>`_) or ``pyproject.toml`` (`pyproject.toml on GitHub <https://github.com/misaghsoltani/NumberLink/blob/main/pyproject.toml>`_)): ``gymnasium``, ``numpy``, ``pygame``.
