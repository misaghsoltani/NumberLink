"""Module entry for the package command-line interface.

Running ``python -m numberlink`` invokes the CLI implemented in :mod:`numberlink.cli`.
"""

from __future__ import annotations

import sys

from .cli import main as cli_main


def run() -> None:
    """Entrypoint used by the ``-m`` switch."""
    raise SystemExit(cli_main(sys.argv[1:]))


if __name__ == "__main__":
    run()
