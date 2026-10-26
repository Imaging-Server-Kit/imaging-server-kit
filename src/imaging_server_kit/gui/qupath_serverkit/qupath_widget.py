from functools import partial
from typing import TYPE_CHECKING, List, Optional

from napari_toolkit.containers.collapsible_groupbox import QCollapsibleGroupBox
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.gui.common.base_widget import BaseRunnerWidget
from imaging_server_kit.gui.qupath_serverkit.qupath_bridge import (
    QuPathBridge,
    if_compatible_get_qupath_schema,
)

if TYPE_CHECKING:
    import napari


def _qupath_compabile_algos(runner: AlgorithmRunner) -> List[str]:
    """Select algorithms from a runner which are Qupath-compatible."""
    compatible_algos = []
    for algo in runner.algorithms:
        if if_compatible_get_qupath_schema(runner, algo)[0]:
            compatible_algos.append(algo)

    return compatible_algos


class LogPanel(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

    def log(self, message):
        self.appendPlainText(f"{message}")


class QuPathWidget(BaseRunnerWidget):
    def __init__(
        self,
        runner: AlgorithmRunner,
        port: int = 25333,
        token: str = "",
        viewer: Optional["napari.Viewer"] = None,
    ):
        super().__init__(runner, algorithms=_qupath_compabile_algos(runner=runner))

        # Optional Napari stack to collect results in at the same time as QuPath
        if viewer is not None:
            from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack

            self.stack = NapariStack(
                viewer, pbar=self.pbar, params_panel=self.params_panel
            )
        else:
            from imaging_server_kit.core import Stack

            self.stack = Stack()

        layout = self.layout()

        # QuPath context (inserted above the runner widget)
        qp_groupbox = QCollapsibleGroupBox("QuPath connection")  # type: ignore
        qp_groupbox.setChecked(False)
        qp_layout = QGridLayout(qp_groupbox)

        qp_layout.addWidget(QLabel("Port"), 0, 0)
        self.port_field = QSpinBox()
        self.port_field.setMaximum(60_000)
        self.port_field.setValue(port)
        qp_layout.addWidget(self.port_field, 0, 1)

        qp_layout.addWidget(QLabel("Token"), 1, 0)
        self.token_field = QLineEdit()
        self.token_field.setText(token)
        qp_layout.addWidget(self.token_field, 1, 1)

        layout.insertWidget(0, qp_groupbox)

        self.qupath_connect_btn = QPushButton("Connect to QuPath")
        self.qupath_connect_btn.clicked.connect(self._connect)
        layout.insertWidget(1, self.qupath_connect_btn)

        layout.insertWidget(2, QLabel("Annotation"))
        self.annotation_field = QComboBox()
        layout.insertWidget(3, self.annotation_field)

        # Logging panel
        self.logger = LogPanel()

        log_groupbox = QCollapsibleGroupBox("Logs")  # type: ignore
        log_groupbox.setChecked(False)
        log_layout = QVBoxLayout(log_groupbox)
        log_layout.addWidget(self.logger)
        log_clear_btn = QPushButton("Clear")
        log_clear_btn.clicked.connect(self.logger.clear)
        log_layout.addWidget(log_clear_btn)
        layout.addWidget(log_groupbox)

        self._select_first_algorithm()

    def _connect(self):
        port = self.port_field.value()
        token = self.token_field.text()

        self.bridge = QuPathBridge()

        try:
            self.bridge.connect(port=port, token=token)
        except Exception:
            self.logger.log("⚠️ Could not connect to QuPath.")
            return

        # If successful, update annotations dropdown
        annotations = self.bridge.get_annotations()
        annotation_names = self.bridge.get_annotation_names(annotations)

        self.annotation_field.clear()
        self.annotation_field.addItems(annotation_names)

        self.logger.log("✅ Connected to QuPath!")

    def _algorithm_changed(self, selected_algo):
        if selected_algo == "":
            return

        try:
            # Check for algo compatibility - if so, retrieve the QuPath-modified schema
            schema, _ = if_compatible_get_qupath_schema(
                self.runner_widget.runner, selected_algo
            )
            if not schema:
                self.logger.log(
                    f"⚠️ Algorithm `{selected_algo}` is incompatible with QuPath."
                )
                return

            # Check if tiled inference should be displayed or not
            algo_is_tileable = self.runner_widget.runner.is_tileable(selected_algo)
            self.params_panel.update(schema)

            self.runner_widget.update_tiled_ui(algo_is_tileable)
        except Exception:
            self.logger.log(f"⚠️ Algorithm `{selected_algo}` is unavailable.")

    def _run(self):
        annotation_name = self.annotation_field.currentText()
        if annotation_name == "":
            self.logger.log("⚠️ Selecting a QuPath annotation is required.")
            return

        algorithm_name = self.runner_widget.selected_algorithm_name
        if algorithm_name == "":
            self.logger.log("⚠️ Selecting an algorithm is required.")
            return

        try:
            qp_annotations = self.bridge.get_annotations()

            found_annotations = self.bridge.get_annotations_by_class_name(
                qp_annotations, cls_name=annotation_name
            )

            if len(found_annotations) == 0:
                self.logger.log(
                    f"⚠️ Could not find the annotation named `{annotation_name}` in QuPath."
                )
                return

            elif len(found_annotations) > 1:
                self.logger.log(
                    f"⚠️ Multiple annotations named `{annotation_name}` found in QuPath (found {len(found_annotations)}). Please give it a unique name."
                )
                return

            annotation = found_annotations[0]

            tiling_ctx = self.runner_widget.selected_tiling_specs

            algo_params = self.params_panel.get_algo_params()

            task = partial(
                self.bridge.run,
                runner=self.runner_widget.runner,
                annotation=annotation,
                tiling_ctx=tiling_ctx,
                algorithm_name=algorithm_name,
                **algo_params,
            )
        except Exception:
            self.logger.log("⚠️ Something went wrong while running the algorithm.")
            return

        self.tasks.add_active(task, self._merge_wrap)

    def _merge_wrap(self, payload):
        if payload is not None:
            result_tile, params_domain = payload
            self.stack.merge(result_tile, reinitialize_domain=params_domain)

            # Update the Qt progress bar
            if result_tile.tile_meta.n_tiles > 0:
                self.pbar.setMaximum(result_tile.tile_meta.n_tiles)
                self.pbar.setValue(result_tile.tile_meta.tile_idx + 1)

            # Update QuPath only on last tile (best solution to handle instance masks)
            if result_tile.tile_meta.is_last_tile:
                self.bridge.merge_with_qupath(self.stack)

    def _log(self, message: str, warning: bool = False) -> None:
        self.logger.log(message)

    def _list_algorithms(self) -> List[str]:
        return _qupath_compabile_algos(runner=self.runner_widget.runner)
