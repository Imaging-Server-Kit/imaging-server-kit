import napari
from imaging_server_kit.gui.napari_serverkit.napari_widget import NapariWidget
from imaging_server_kit.gui.common import HttpRunnerWidget


class NapariHttpWidget(NapariWidget):
    def __init__(self, viewer: napari.Viewer):
        runner_widget = HttpRunnerWidget()
        super().__init__(runner_widget=runner_widget, viewer=viewer)
