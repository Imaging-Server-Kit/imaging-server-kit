from functools import partial
from typing import Dict, Optional

from napari_toolkit.containers.collapsible_groupbox import QCollapsibleGroupBox
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.gui.common import ParameterPanel, RunnerWidget, TaskManager
from imaging_server_kit.gui.qupath_serverkit.qupath_bridge import QuPathBridge


def _if_compatible_get_qupath_schema(runner: AlgorithmRunner, algorithm: str) -> Dict:
    """Check if an algorithm is compatible with a QuPath run (= single image as input + no masks, points, etc.)."""
    schema = runner.get_parameters(algorithm)
    params = schema["properties"]

    image_params = []
    for param_name, param in params.items():
        # Parameters inputs that QuPath won't support (if they are required):
        if (param["required"] is True) and (
            param["param_type"]
            in [
                "mask",
                "paths",
                "boxes",
                "points",
                "vectors",
                "tracks",
            ]
        ):
            param_type = param["param_type"]
            print(
                f"Algorithm `{algorithm}` is incompatible with QuPath (requires a parameter of type `{param_type}` as input)."
            )
            return {}

        if param["param_type"] == "image":
            image_params.append(param_name)

    if len(image_params) == 1:
        qp_image_param = image_params[0]
        params.pop(qp_image_param)
        schema["properties"] = params

        return schema
    else:
        print(
            f"Algorithm `{algorithm}` is incompatible with QuPath (requires `{len(image_params)}` images as input)."
        )
        return {}


class LogPanel(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

    def log(self, message):
        self.appendPlainText(f"{message}")


class QuPathWidget(QWidget):
    def __init__(
        self,
        runner_widget: RunnerWidget,
        port: int = 25333,
        token: str = "",
        viewer: Optional["napari.Viewer"] = None,
    ):
        super().__init__()

        # Create a dynamic parameters panel
        self.params_panel = ParameterPanel(trigger=self._run)

        # Progress bar
        self.pbar = QProgressBar(minimum=0, maximum=1)  # type: ignore

        # Optional Napari stack to collect results in at the same time as QuPath
        if viewer is not None:
            from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack

            self.napari_stack = NapariStack(
                viewer, pbar=self.pbar, params_panel=self.params_panel
            )
        else:
            self.napari_stack = None

        # Runner widget
        self.runner_widget = runner_widget

        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)  # type: ignore
        self.setLayout(layout)

        # QuPath context
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

        layout.addWidget(qp_groupbox)

        self.qupath_connect_btn = QPushButton("Connect to QuPath")
        self.qupath_connect_btn.clicked.connect(self._connect)
        layout.addWidget(self.qupath_connect_btn)

        layout.addWidget(QLabel("Annotation"))
        self.annotation_field = QComboBox()
        layout.addWidget(self.annotation_field)

        # Add the runner's extra UI
        layout.addWidget(self.runner_widget.widget)

        # Connect the ComboBox change from the runner to the UI update
        self.runner_widget.update_params_trigger.connect(self._algorithm_changed)

        # Add the parameters panel
        layout.addWidget(self.params_panel.widget)

        # Run button
        self.run_btn = QPushButton("Run", self)
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        # Task manager
        self.tasks = TaskManager(
            self._grayout_ui,  # called when worker starts
            self._ungrayout_ui,  # called when worker stops
            self.params_panel,  # linked to manage_cbs_events(worker)
        )

        self.grayout_ui_list = [
            self.params_panel.widget,
            self.run_btn,
            self.runner_widget.widget,
        ]

        # Cancel button
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(cancel_btn)

        # Progress bar (= soon to be its own layout appearing at the bottom)
        layout.addWidget(self.pbar)

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

        # Trigger an initial `algorithm selection`
        if len(self.runner_widget.runner.algorithms) > 0:
            self._algorithm_changed(self.runner_widget.runner.algorithms[0])

    def _connect(self):
        port = self.port_field.value()
        token = self.token_field.text()

        self.bridge = QuPathBridge()

        try:
            self.bridge.connect(port=port, token=token)
        except:
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
            # Check for algo compatibility - if so, retreive the QuPath-modified schema
            schema = _if_compatible_get_qupath_schema(
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
        except:
            self.logger.log("⚠️ Algorithm `{selected_algo}` is unavailable.")

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
                    f"⚠️ Multiple annotations named `{annotation_name}` found in QuPath (found {len(found_annotations)}). Using the first one."
                )
                return

            annotation = found_annotations[0]

            tiling_ctx = self.runner_widget.selected_tiling_specs

            algo_params = self.params_panel.get_algo_params()

            task = partial(
                self.bridge.run_in_annotation,
                runner=self.runner_widget.runner,
                annotation=annotation,
                tiling_ctx=tiling_ctx,
                algorithm_name=algorithm_name,
                **algo_params,
            )
        except:
            self.logger.log("⚠️ Something went wrong while running the algorithm.")
            return

        self.tasks.add_active(task, self._merge_wrap)

    def _merge_wrap(self, payload):
        if payload is not None:
            result_tile, params_domain = payload
            if self.napari_stack is not None:
                # Update Napari
                self.napari_stack.merge(result_tile, reinitialize_domain=params_domain)
            else:
                # Update the Qt progress bar
                if result_tile.tile_meta.n_tiles > 0:
                    self.pbar.setMaximum(result_tile.tile_meta.n_tiles)
                    self.pbar.setValue(result_tile.tile_meta.tile_idx + 1)

            # Update QuPath
            self.bridge.merge_with_qupath(result_tile)

    def _cancel(self):
        self.logger.log("⏳ Cancelling...")
        self.tasks.cancel_all()

    def _aborted(self):
        self._ungrayout_ui()
        self.pbar.setMaximum(1)

    def _grayout_ui(self):
        self.pbar.setMaximum(0)  # Start the pbar
        for ui_element in self.grayout_ui_list:
            ui_element.setEnabled(False)

    def _ungrayout_ui(self):
        self.pbar.setMaximum(1)  # Stop the pbar
        for ui_element in self.grayout_ui_list:
            ui_element.setEnabled(True)
