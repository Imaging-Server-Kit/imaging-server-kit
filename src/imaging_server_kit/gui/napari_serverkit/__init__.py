from typing import Callable, Optional, Union
import importlib.util

from imaging_server_kit import Algorithm
from imaging_server_kit.core.runner import AlgorithmRunner


def napari_available() -> bool:
    return importlib.util.find_spec("napari") is not None


def to_qwidget(
    runner: Union[AlgorithmRunner, Callable], viewer: "napari.Viewer"
) -> "QWidget":
    """Convert an algorithm to a QWidget. Used when packaging a Napari plugin."""
    if not napari_available():
        raise ImportError(
                """
                    This function requires the optional Napari dependencies to be installed.\n
                    Install them with: `pip install imaging-server-kit[napari]`.
                """
            )
    
    from .napari_widget import NapariWidget
    
    if not isinstance(runner, AlgorithmRunner):
        runner = Algorithm(runner)

    return NapariWidget(viewer=viewer, runner=runner)


def to_napari(
    runner: Union[AlgorithmRunner, Callable],
    viewer: Optional["napari.Viewer"] = None,
) -> "napari.Viewer":
    """
    Convert an algorithm (or algorithm collection) to a dock widget and add it to a Napari viewer.

    Parameters
    ----------
    algorithm : The algorithm object to add to Napari as a dock widget.
    viewer : An existing Napari viewer to add the dock widget to. If none is passed, a new Napari viewer is created.
    """
    if not napari_available():
        raise ImportError(
                """
                    This function requires the optional Napari dependencies to be installed.\n
                    Install them with: `pip install imaging-server-kit[napari]`.
                """
            )

    import napari
    
    if not isinstance(runner, AlgorithmRunner):
        runner = Algorithm(run_algorithm_func=runner)

    if viewer is None:
        viewer = napari.Viewer()

    widget = to_qwidget(runner=runner, viewer=viewer)

    viewer.window.add_dock_widget(widget=widget, name=runner.name)

    return viewer




