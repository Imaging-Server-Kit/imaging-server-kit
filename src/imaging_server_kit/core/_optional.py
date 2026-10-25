"""Checks for the optional dependencies (extras) of the package."""

import importlib.util


def napari_available() -> bool:
    return importlib.util.find_spec("napari") is not None


def qubalab_available() -> bool:
    return importlib.util.find_spec("qubalab") is not None


def require_napari() -> None:
    """Raise an ImportError if the `napari` extra is not installed."""
    if not napari_available():
        raise ImportError(_missing_extra_message("Napari", "napari"))


def require_qupath() -> None:
    """Raise an ImportError if the `qupath` extra is not installed."""
    if not qubalab_available():
        raise ImportError(_missing_extra_message("QuPath", "qupath"))


def _missing_extra_message(name: str, extra: str) -> str:
    return (
        f"This function requires the optional {name} dependencies to be installed.\n"
        f"Install them with: `pip install imaging-server-kit[{extra}]`."
    )
