from typing import List, Optional

from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.gui.common import RunnerWidget

from .qupath_widget import QuPathWidget, _if_compatible_get_qupath_schema


def _qupath_compabile_algos(runner: AlgorithmRunner) -> List[str]:
    """Select algorithms from a runner which are Qupath-compatible."""
    compatible_algos = []
    for algo in runner.algorithms:
        if _if_compatible_get_qupath_schema(runner, algo):
            compatible_algos.append(algo)
    
    return compatible_algos


class QuPathAlgorithmWidget(QuPathWidget):
    def __init__(
        self,
        port: int,
        token: str,
        runner: AlgorithmRunner,
        viewer: Optional["napari.Viewer"] = None,
    ):
        # Select algorithms compatible with QuPath
        qupath_compatible_algos = _qupath_compabile_algos(runner=runner)
        runner_widget = RunnerWidget(runner=runner, algorithms=qupath_compatible_algos)
        super().__init__(
            port=port,
            token=token,
            runner_widget=runner_widget,
            viewer=viewer,
        )
