from typing import Union
from ._version import version as __version__

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
)

from .merge import merge_layers, LayerMerger

from .demo import multi_algo_tools as tools
from .demo import multi_algo_demos as demos

from .remote import Client, serve
from .gui import to_napari, to_qwidget


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
        from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack

        # For napari, we return the viewer directly
        napari_stack = NapariStack(layers=stack.layers)
        return napari_stack.viewer
