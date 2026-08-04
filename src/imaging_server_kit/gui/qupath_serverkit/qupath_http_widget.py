from typing import Optional

from imaging_server_kit.gui.common import HttpRunnerWidget

from .qupath_widget import QuPathWidget

import napari
class QuPathHttpWidget(QuPathWidget):
    def __init__(
        self,
        viewer: napari.Viewer,  # TODO: it has to be this, apparently...
        # viewer: Optional["napari.Viewer"],
        port: int = 25333,
        token: str = "",
    ):
        runner_widget = HttpRunnerWidget()
        super().__init__(
            port=port,
            token=token,
            runner_widget=runner_widget,
            viewer=viewer,
        )
