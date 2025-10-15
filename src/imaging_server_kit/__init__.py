from typing import Optional, Union, Callable
from ._version import version as __version__

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core import (
    Client,
    Algorithm,
    Parameters,
    generate_nd_tiles,
    Results,
    LayerStackBase,
    algorithm,
    MultiAlgorithm,
    combine,
)

from .types import (
    Image,
    Mask,
    Points,
    Vectors,
    Boxes,
    Paths,
    Tracks,
    Float,
    Integer,
    Bool,
    String,
    DropDown,
    Notification,
)

from .core.errors import napari_available

NAPARI_INSTALLED = napari_available()


def to_qwidget(algorithm: Optional[Union[Algorithm, MultiAlgorithm, Callable]] = None):
    """Convert an algorithm to a QWidget."""
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

    widget = AlgorithmWidget(viewer=None, server=algorithm)

    return widget


def to_napari(
    algorithm: Optional[Union[Algorithm, MultiAlgorithm, Callable]] = None,
    viewer: Optional[Union["napari.Viewer", "napari_serverkit.NapariResults"]] = None,
):
    """Convert an algorithm to a dock widget and add it to a napari viewer."""
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


def serve(algorithm: Union[Algorithm, MultiAlgorithm, Callable], *args, **kwargs):
    """Serve an algorithm as an HTTP server."""
    try:
        from imaging_server_kit.core.app import AlgorithmApp
    except ImportError:
        print(
            "To use this method, install the Imaging Server Kit with the server extension: `pip install imaging-server-kit[server]`."
        )
        return

    if isinstance(algorithm, Algorithm):
        algorithm_servers = [algorithm]
    elif isinstance(algorithm, MultiAlgorithm):
        algorithm_servers = list(algorithm.algorithms_dict.values())
    else:
        # Assuming the user has passed a "raw" Python function, we attempt to convert it to an Algorithm:
        algorithm = Algorithm(algorithm)
        algorithm_servers = [algorithm]

    algo_app = AlgorithmApp(algorithm.name, algorithm_servers=algorithm_servers)
    algo_app.serve(*args, **kwargs)


def convert(results: LayerStackBase, to: str = "results") -> Union[LayerStackBase, "napari.Viewer"]:
    """Convert a set of results to various respresentations."""
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
            kind=layer.kind, data=layer.data, name=layer.name, meta=layer.meta
        )

    if to == "napari":
        # For napari specifically, return a reference to the viewer so that users can call add_image on it instead of create()
        # + When using results = runner.run(results=viewer), we should return the viewer
        return results_dst.viewer
    else:
        return results_dst
