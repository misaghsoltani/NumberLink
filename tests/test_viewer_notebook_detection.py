import sys
from types import ModuleType

from pytest import MonkeyPatch

from numberlink.viewer import _detect_notebook_environment  # noqa: PLC2701


def test_detect_notebook_environment_colab_without_zmq(monkeypatch: MonkeyPatch) -> None:
    """Test detection of Google Colab environment without zmq installed."""

    class DummyShell:
        kernel = object()

    shell = DummyShell()

    getipython_mod = ModuleType("IPython.core.getipython")
    getipython_mod.get_ipython = lambda: shell

    core_mod = ModuleType("IPython.core")
    core_mod.getipython = getipython_mod

    ipython_mod = ModuleType("IPython")
    ipython_mod.core = core_mod

    monkeypatch.setitem(sys.modules, "IPython", ipython_mod)
    monkeypatch.setitem(sys.modules, "IPython.core", core_mod)
    monkeypatch.setitem(sys.modules, "IPython.core.getipython", getipython_mod)

    monkeypatch.setitem(sys.modules, "google.colab", ModuleType("google.colab"))
    monkeypatch.setenv("COLAB_RELEASE_TAG", "1.0")

    assert _detect_notebook_environment() == "colab"
