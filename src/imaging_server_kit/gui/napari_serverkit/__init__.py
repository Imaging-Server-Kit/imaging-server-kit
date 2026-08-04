from typing import Callable, Optional, Union

from qtpy.QtWidgets import QWidget

import napari

from imaging_server_kit import Algorithm, tools, demos
from imaging_server_kit.core.runner import AlgorithmRunner

from imaging_server_kit.remote import Client

from .napari_algo_widget import NapariAlgorithmWidget
from .napari_http_widget import NapariHttpWidget
from .napari_stack import NapariStack


def to_qwidget(
    runner: Union[AlgorithmRunner, Callable], viewer: napari.Viewer
) -> QWidget:
    """Convert an algorithm to a QWidget. Used when packaging a Napari plugin."""
    if not isinstance(runner, AlgorithmRunner):
        runner = Algorithm(runner)

    if isinstance(runner, Client):
        return NapariHttpWidget(viewer=viewer)
    else:
        return NapariAlgorithmWidget(viewer=viewer, runner=runner)


def to_napari(
    runner: Union[AlgorithmRunner, Callable],
    viewer: Optional[napari.Viewer] = None,
) -> napari.Viewer:
    """
    Convert an algorithm (or algorithm collection) to a dock widget and add it to a Napari viewer.

    Parameters
    ----------
    algorithm : The algorithm object to add to Napari as a dock widget.
    viewer : An existing Napari viewer to add the dock widget to. If none is passed, a new Napari viewer is created.
    """
    if not isinstance(runner, AlgorithmRunner):
        runner = Algorithm(run_algorithm_func=runner)

    if viewer is None:
        viewer = napari.Viewer()

    widget = to_qwidget(runner=runner, viewer=viewer)

    viewer.window.add_dock_widget(widget=widget, name=runner.name)

    return viewer


class AlgorithmToolsWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        widget = to_qwidget(tools, viewer=viewer)
        self.setLayout(widget.layout())


class AlgorithmDemosWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        widget = to_qwidget(demos, viewer=viewer)
        self.setLayout(widget.layout())
