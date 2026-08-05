import napari
from imaging_server_kit.gui.napari_serverkit.napari_widget import NapariWidget
from imaging_server_kit.remote import Client


class NapariHttpWidget(NapariWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer=viewer, runner=Client())
