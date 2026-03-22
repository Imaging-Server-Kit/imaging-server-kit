from typing import Optional, Union, Callable
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
    LayerStackBase,
    Results,
    generate_tiles,
    TileMeta,
)

from .remote import Client, serve

from .types import (
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

from .core.errors import napari_available

NAPARI_INSTALLED = napari_available()


def to_qwidget(algorithm: Optional[Union[Algorithm, MultiAlgorithm, Callable]], viewer):
    """Convert an algorithm to a QWidget. Used when packaging a Napari plugin."""
    if not NAPARI_INSTALLED:
        print(
            "To use this method, install the Imaging Server Kit Napari plugin with `pip install napari-serverkit`."
        )
        return

    from napari_serverkit import AlgorithmWidget

    if algorithm is not None:
        if not isinstance(algorithm, (Algorithm, MultiAlgorithm)):
            # Assuming the user has passed a "raw" Python function, we attempt to convert it to an Algorithm:
            algorithm = Algorithm(algorithm)

    return AlgorithmWidget(viewer=viewer, algorithm=algorithm)


def to_napari(
    algorithm: Optional[Union[Algorithm, MultiAlgorithm, Callable]] = None,
    viewer: Optional[Union["napari.Viewer", "napari_serverkit.NapariResults"]] = None,
) -> None:
    """
    Convert an algorithm (or algorithm collection) to a dock widget and add it to a Napari viewer.

    Parameters
    ----------
    algorithm : The algorithm object to add to Napari as a dock widget.
    viewer : An existing Napari viewer to add the dock widget to. If none is passed, a new Napari viewer is created.
    """
    if not NAPARI_INSTALLED:
        print(
            "To use this method, install the Imaging Server Kit Napari plugin with `pip install napari-serverkit`."
        )
        return

    import napari
    from napari_serverkit import add_as_widget

    if viewer is None:
        viewer = napari.Viewer()

    if algorithm is not None:
        if not isinstance(algorithm, (Algorithm, MultiAlgorithm)):
            # Assuming the user has passed a "raw" Python function, we attempt to convert it to an Algorithm:
            algorithm = Algorithm(algorithm)

        add_as_widget(viewer, algorithm)

    return viewer


def convert(
    results: LayerStackBase, to: str = "results"
) -> Union[LayerStackBase, "napari.Viewer"]:
    """
    Convert a result object into a different representation.

    Parameters
    ----------
    results : The result object to convert.
    to : The target representation to convert to. Supported values: ["results", "napari"]

    Returns
    -------
    The converted result object.
    - If `to == "results"`, a Results() object containing copies of the input layers.
    - If `to == "napari"` the napari.Viewer associated with the converted results.
    """
    supported_results = ["results", "napari"]
    if not to in supported_results:
        raise ValueError(f"{to} is not supported. Please use {supported_results}")

    if to == "results":
        results_dst = Results()
    elif to == "napari":
        if not NAPARI_INSTALLED:
            print(
                "To use this method, install the Imaging Server Kit Napari plugin with `pip install napari-serverkit`."
            )
            return
        from napari_serverkit import NapariResults

        results_dst = NapariResults()

    for layer in results:
        results_dst.create(
            kind=layer.kind,
            data=layer.data,
            name=layer.name,
            meta=layer.meta,
        )

    if to == "napari":
        # For napari, we return the viewer directly
        return results_dst.viewer
    else:
        return results_dst
