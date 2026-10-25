from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QProgressBar, QPushButton, QVBoxLayout, QWidget

from imaging_server_kit.core.errors import AlgorithmServerError, ServerRequestError
from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.gui.common.parameter_panel import ParameterPanel
from imaging_server_kit.gui.common.runner_widget import RunnerWidget
from imaging_server_kit.gui.common.task_manager import TaskManager
from imaging_server_kit.remote.client import Client


class BaseRunnerWidget(QWidget):
    """Layout and behaviour shared by the Napari and QuPath widgets.

    Builds the runner selection, parameters panel, Run and Cancel buttons and progress bar.
    Subclasses implement `_algorithm_changed()`, `_run()` and `_log()`, can override
    `_list_algorithms()`, and call `_select_first_algorithm()` at the end of their `__init__`.
    """

    def __init__(self, runner: AlgorithmRunner, algorithms: Optional[List[str]] = None):
        super().__init__()

        # Algorithm parameters (dynamic UI)
        self.params_panel = ParameterPanel(trigger=self._run)

        # Progress bar
        self.pbar = QProgressBar(minimum=0, maximum=1)  # type: ignore

        # Runner widget
        self.runner_widget = RunnerWidget(runner, algorithms=algorithms)

        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)

        # Add the runner's extra UI
        layout.addWidget(self.runner_widget)

        # Connect the server URL field
        self.runner_widget.connect_btn.clicked.connect(self._connect_from_btn)

        # Connect the ComboBox change from the runner to the UI update
        self.runner_widget.cb_algorithms.currentTextChanged.connect(
            self._algorithm_changed
        )

        # Connect the doc link button
        self.runner_widget.algo_info_btn.clicked.connect(self._open_info_link_from_btn)

        # Add the parameters panel
        layout.addWidget(self.params_panel)

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

        self.grayout_ui_list = [self.params_panel, self.run_btn, self.runner_widget]

        # Cancel button
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(cancel_btn)

        # Progress bar (= soon to be its own layout appearing at the bottom)
        layout.addWidget(self.pbar)

    def _select_first_algorithm(self) -> None:
        """Trigger an initial `algorithm selection`."""
        if len(self.runner_widget.runner.algorithms) > 0:
            self._algorithm_changed(self.runner_widget.runner.algorithms[0])

    def _algorithm_changed(self, selected_algo: str) -> None:
        raise NotImplementedError

    def _run(self) -> None:
        raise NotImplementedError

    def _log(self, message: str, warning: bool = False) -> None:
        """Show a message to the user."""
        raise NotImplementedError

    def _list_algorithms(self) -> List[str]:
        """Algorithms listed in the dropdown after connecting to a server."""
        return self.runner_widget.runner.algorithms

    def _open_info_link_from_btn(self, *args, **kwargs) -> None:
        algorithm = self.runner_widget.selected_algorithm_name
        if algorithm == "":
            return

        self.runner_widget.runner.info(algorithm)

    def _connect_from_btn(self):
        server_url = self.runner_widget.server_url_field.text()
        if server_url == "":
            self._log("Please specify a server URL!", warning=True)
            return

        if isinstance(self.runner_widget.runner, Client):
            try:
                self.runner_widget.runner.connect(server_url)
            except (ServerRequestError, AlgorithmServerError):
                self._log(f"Could not connect to server on {server_url}", warning=True)

        self.runner_widget.cb_algorithms.clear()
        self.runner_widget.cb_algorithms.addItems(self._list_algorithms())

    def _cancel(self):
        self._log("⏳ Cancelling...")
        self.tasks.cancel_all()

    def _grayout_ui(self):
        self.pbar.setMaximum(0)  # Start the pbar
        for ui_element in self.grayout_ui_list:
            ui_element.setEnabled(False)

    def _ungrayout_ui(self):
        self.pbar.setMaximum(1)  # Stop the pbar
        for ui_element in self.grayout_ui_list:
            ui_element.setEnabled(True)
