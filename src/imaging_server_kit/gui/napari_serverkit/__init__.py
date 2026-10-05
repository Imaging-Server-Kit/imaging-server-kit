from typing import Callable, Optional, Union
import importlib.util

from imaging_server_kit import Algorithm
from imaging_server_kit.core.runner import AlgorithmRunner


def napari_available() -> bool:
    return importlib.util.find_spec("napari") is not None


def to_qwidget(
    runner: Union[AlgorithmRunner, Callable], viewer: "napari.Viewer"
) -> "QWidget":
    """Create the Napari widget of an algorithm, collection, or client, without adding it to a viewer.

    Useful to ship an algorithm as a Napari plugin of its own. Requires the `napari`
    extra.

    Parameters
    ----------
    runner : AlgorithmRunner or callable
        An algorithm, an algorithm collection, or a client. A plain Python function is
        converted to an algorithm.
    viewer : napari.Viewer
        The Napari viewer the widget interacts with.

    Returns
    -------
    QWidget
        The widget, exposing the parameters and results of the runner.
    """
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
    """Add the dock widget of an algorithm, collection, or client to a Napari viewer.

    Requires the `napari` extra.

    Parameters
    ----------
    runner : AlgorithmRunner or callable
        An algorithm, an algorithm collection, or a client. A plain Python function is
        converted to an algorithm.
    viewer : napari.Viewer, optional
        The viewer to add the widget to. By default, a new viewer is created.

    Returns
    -------
    napari.Viewer
        The viewer with the dock widget.

    Examples
    --------
    >>> viewer = sk.to_napari(threshold_algo)
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




