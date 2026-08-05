import napari
from imaging_server_kit.gui.qupath_serverkit.qupath_widget import QuPathWidget
from imaging_server_kit.remote import Client


class QuPathHttpWidget(QuPathWidget):
    def __init__(self, viewer: napari.Viewer, port: int = 25333, token: str = ""):
        super().__init__(viewer=viewer, port=port, token=token, runner=Client())
