from typing import Optional, Union

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core import (
    algorithm,
    Algorithm,
    MultiAlgorithm,
    combine,
    Stack,
    generate_tiles,
    TileMeta,
    Domain,
)
from .core.runner import AlgorithmRunner

from .types import (
    Layer,
    Image,
    Mask,
    Paths,
    Boxes,
    Points,
    Vectors,
    Tracks,
    Float,
    Integer,
    Bool,
    String,
    Choice,
    Notification,
    Null,
    Progress,
    Any,
)

from .merge import merge_layers, LayerMerger

from .demo import multi_algo_tools as tools
from .demo import multi_algo_demos as demos

from .remote import Client, serve
from .gui import to_napari, to_qwidget, to_qupath


def convert(stack: Stack, to: str = "stack") -> Union[Stack, "napari.Viewer"]:
    """
    Convert a result object into a different representation.

    Parameters
    ----------
    stack : The result object to convert.
    to : The target representation to convert to. Supported values: ["stack", "napari"]

    Returns
    -------
    The converted result object.
    - If `to == "stack"`, a Stack() object containing copies of the input layers.
    - If `to == "napari"` the napari.Viewer associated with the converted stack.
    """
    supported = ["stack", "napari"]
    if not to in supported:
        raise ValueError(f"{to} is not supported. Please use {supported}")

    if to == "stack":
        return Stack(layers=stack.layers)
    elif to == "napari":
        from imaging_server_kit.gui.napari_serverkit import napari_available

        if not napari_available():
            raise ImportError("""
                    This function requires the optional Napari dependencies to be installed.\n
                    Install them with: `pip install imaging-server-kit[napari]`.
                """)

        from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack

        # For napari, we return the viewer directly
        napari_stack = NapariStack(layers=stack.layers)
        return napari_stack.viewer


def run(
    runner: AlgorithmRunner,
    *args,
    algorithm: Optional[str] = None,
    tiled: bool = False,
    tile_size: int = 64,
    tile_overlap: float = 0.0,
    tile_delay: float = 0.0,
    tile_randomize: bool = False,
    stack: Union[Stack, "napari.Viewer"] = None,  # type: ignore
    domain: Optional[Domain] = None,
    **algo_params,
) -> Union[Stack, "napari.Viewer"]:  # type: ignore
    """Allows the syntax `sk.run(...)` instead of runner.run(...)."""
    return runner.run(
        *args,
        algorithm=algorithm,
        tiled=tiled,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_delay=tile_delay,
        tile_randomize=tile_randomize,
        stack=stack,
        domain=domain,
        **algo_params,
    )


def info(runner: AlgorithmRunner, algorithm: Optional[str] = None):
    """Allow the syntax `sk.info(...)` instead of runner.info(...)."""
    return runner.info(algorithm=algorithm)
