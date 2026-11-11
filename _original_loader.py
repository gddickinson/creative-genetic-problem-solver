"""Load the hyphenated original scripts (e.g. ``flow-network.py``) as importable modules.

The underscore modules (``flow_network.py`` ...) are thin wrappers around this loader, so there is one copy of each
solver's code. ``load`` registers the module in ``sys.modules`` under ``<stem>_orig`` (loaded once); ``run_as_main``
runs the original as a script, so ``python flow_network.py`` behaves like ``python flow-network.py``.
"""
import importlib.util
import os
import runpy
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(filename: str) -> str:
    return os.path.join(_DIR, filename)


def load(filename: str):
    """Import the original script ``filename`` (hyphenated, in the repo root) and return the module."""
    name = os.path.splitext(filename)[0].replace("-", "_") + "_orig"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _path(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]
        raise
    return mod


def run_as_main(filename: str) -> None:
    """Run the original script as ``__main__`` (its interactive desktop entry point)."""
    runpy.run_path(_path(filename), run_name="__main__")
