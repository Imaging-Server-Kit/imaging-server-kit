import napari
from qtpy.QtWidgets import QWidget

from imaging_server_kit import tools, demos
from imaging_server_kit.gui.napari_serverkit import to_qwidget


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