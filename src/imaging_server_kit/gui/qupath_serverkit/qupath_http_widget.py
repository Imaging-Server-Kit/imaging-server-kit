import napari
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from imaging_server_kit.core._optional import require_qupath
from imaging_server_kit.remote import Client


class QuPathHttpWidget(QWidget):
    """QuPath panel of the Napari plugin menu, connected to a server.

    The module must import without the `qupath` extra: Napari imports every widget of the
    plugin, e.g. on `napari -w imaging-server-kit`. Without the extra, the panel shows how
    to install it.
    """

    def __init__(self, viewer: napari.Viewer, port: int = 25333, token: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            require_qupath()
        except ImportError as e:
            label = QLabel(str(e))
            label.setWordWrap(True)
            layout.addWidget(label)
            return

        from imaging_server_kit.gui.qupath_serverkit.qupath_widget import QuPathWidget

        self.qupath_widget = QuPathWidget(viewer=viewer, port=port, token=token, runner=Client())
        layout.addWidget(self.qupath_widget)
