from __future__ import annotations
import sys
from typing import Callable, Optional, Union, TYPE_CHECKING

from imaging_server_kit.core._optional import require_qupath
from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.algorithm import as_runner

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget
    from napari import Viewer


def to_qwidget(
    runner: Union[AlgorithmRunner, Callable],
    port: int = 25333,
    token: str = "",
    viewer=None,
) -> QWidget:
    """Create the QuPath panel of an algorithm, collection, or client, as a QWidget.

    Parameters
    ----------
    runner : AlgorithmRunner or callable
        An algorithm, an algorithm collection, or a client. A plain Python function is
        converted to an algorithm.
    port : int, default=25333
        Port of the Py4J gateway started from QuPath.
    token : str, default=""
        Token of the Py4J gateway started from QuPath.
    viewer : napari.Viewer, optional
        A Napari viewer to collect results that cannot be displayed in QuPath.

    Returns
    -------
    QWidget
        The QuPath panel.
    """
    require_qupath()
    
    from .qupath_widget import QuPathWidget
    
    runner = as_runner(runner)

    return QuPathWidget(port=port, token=token, runner=runner, viewer=viewer)


def to_qupath(
    runner: Union[AlgorithmRunner, Callable],
    port: int = 25333,
    token: str = "",
    viewer: Optional[Viewer] = None,
) -> Optional[Viewer]:
    """Open a panel to run an algorithm, collection, or client on QuPath images. Experimental.

    The panel connects to QuPath through QuBaLab and a Py4J gateway started from
    QuPath. Computations run inside a selected QuPath annotation (e.g. a rectangular
    region). Only algorithms that take a single image as input are compatible; that
    image is interpreted as the current QuPath image. Requires the `qupath` extra.

    Parameters
    ----------
    runner : AlgorithmRunner or callable
        An algorithm, an algorithm collection, or a client. A plain Python function is
        converted to an algorithm.
    port : int, default=25333
        Port of the Py4J gateway started from QuPath.
    token : str, default=""
        Token of the Py4J gateway started from QuPath.
    viewer : napari.Viewer, optional
        A Napari viewer to collect results that cannot be displayed in QuPath. If
        given, the panel is added to the viewer as a dock widget, and the viewer is
        returned. Otherwise, the panel opens in its own window.

    Returns
    -------
    napari.Viewer or None
        The viewer, if one was given.
    """
    require_qupath()
    
    runner = as_runner(runner)

    if viewer is not None:
        widget = to_qwidget(runner=runner, port=port, token=token, viewer=viewer)
        viewer.window.add_dock_widget(widget)
        return viewer
    else:
        from qtpy.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        widget = to_qwidget(runner=runner, port=port, token=token)
        widget.show()
        sys.exit(app.exec())
