from typing import Callable, Optional, Union

from imaging_server_kit.core._optional import require_napari
from imaging_server_kit.core.algorithm import as_runner
from imaging_server_kit.core.runner import AlgorithmRunner


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
    require_napari()
    
    from .napari_widget import NapariWidget
    
    runner = as_runner(runner)

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
    require_napari()

    import napari
    
    runner = as_runner(runner)

    if viewer is None:
        viewer = napari.Viewer()

    widget = to_qwidget(runner=runner, viewer=viewer)

    viewer.window.add_dock_widget(widget=widget, name=runner.name)

    return viewer




