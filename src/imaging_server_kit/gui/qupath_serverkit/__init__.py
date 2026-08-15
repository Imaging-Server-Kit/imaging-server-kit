import sys
from typing import Callable, Optional, Union
import importlib.util

from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.algorithm import Algorithm


def qubalab_available() -> bool:
    return importlib.util.find_spec("qubalab") is not None


def to_qwidget(
    runner: Union[AlgorithmRunner, Callable],
    port: int = 25333,
    token: str = "",
    viewer=None,
) -> "QWidget":
    """Convert an algorithm to a QWidget (QuPath version)."""
    if not qubalab_available():
        raise ImportError(
                """
                    This function requires the optional QuPath dependencies to be installed.\n
                    Install them with: `pip install imaging-server-kit[qupath]`.
                """
            )
    
    from .qupath_widget import QuPathWidget
    
    if not isinstance(runner, AlgorithmRunner):
        runner = Algorithm(runner)

    return QuPathWidget(port=port, token=token, runner=runner, viewer=viewer)


def to_qupath(
    runner: Union[AlgorithmRunner, Callable],
    port: int = 25333,
    token: str = "",
    viewer: Optional["napari.Viewer"] = None,
) -> None:
    """
    Experimental bridge between QuPath and the Imaging Server Kit via QuBaLab.

    This function creates a user interface for a server kit algorithm (or multi-algorithm, or client)
    which can be used to run computations inside a QuPath annotation (e.g., a selected rectangular region).

    Only algorithms that take a single image as input are compatible (this image is interpreted as the QuPath image).

    Parameters
    ----------
    runner: A server kit algorithm, multi-algorithm, or client object.
    port: Port from the Py4J extension.
    token: Token from the Py4J extension.
    viewer: An optional Napari Viewer to use to collect results from the compuatations that cannot be displayed in QuPath.
    """
    if not qubalab_available():
        raise ImportError(
                """
                    This function requires the optional QuPath dependencies to be installed.\n
                    Install them with: `pip install imaging-server-kit[qupath]`.
                """
            )
    
    if not isinstance(runner, AlgorithmRunner):
        runner = Algorithm(run_algorithm_func=runner)

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
