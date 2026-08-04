import napari
from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.gui.common import RunnerWidget
from imaging_server_kit.gui.napari_serverkit.napari_widget import NapariWidget


class NapariAlgorithmWidget(NapariWidget):
    def __init__(self, viewer: napari.Viewer, runner: AlgorithmRunner):
        runner_widget = RunnerWidget(runner)
        super().__init__(runner_widget=runner_widget, viewer=viewer)
