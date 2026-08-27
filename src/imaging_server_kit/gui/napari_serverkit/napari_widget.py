from functools import partial

import napari
from napari.utils.notifications import show_info, show_warning
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QProgressBar, QPushButton, QVBoxLayout, QWidget

import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.errors import AlgorithmServerError
from imaging_server_kit.core.stack import Stack
from imaging_server_kit.gui.common import (
    NAPARI_LAYER_TYPES,
    ParameterPanel,
    RunnerWidget,
    TaskManager,
)
from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack
from imaging_server_kit.types import layer_factory
from imaging_server_kit.remote.client import Client, ServerRequestError
from imaging_server_kit.core.errors import AlgorithmServerError
from imaging_server_kit.core.runner import AlgorithmRunner


class NapariWidget(QWidget):
    def __init__(self, runner: AlgorithmRunner, viewer: napari.Viewer):
        super().__init__()

        # Algorithm parameters (dynamic UI)
        self.params_panel = ParameterPanel(trigger=self._run)

        # Progress bar (can be accessed by NapariStack; will eventually turn into a "Stack" object)
        self.pbar = QProgressBar(minimum=0, maximum=1)  # type: ignore

        # Shared with NapariStack here...
        self.napari_stack = NapariStack(
            viewer, pbar=self.pbar, params_panel=self.params_panel
        )

        # # Runner widget
        self.runner_widget = RunnerWidget(runner)

        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)  # type: ignore
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

        # Connect the samples loading event
        self.runner_widget.samples_select_btn.clicked.connect(self._sample_triggered)

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

        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self._cancel)
        layout.addWidget(cancel_btn)

        # Progress bar (= soon to be its own layout appearing at the bottom)
        layout.addWidget(self.pbar)

        # Trigger an initial `algorithm selection`
        if len(self.runner_widget.runner.algorithms) > 0:
            self._algorithm_changed(self.runner_widget.runner.algorithms[0])
            
    def _algorithm_changed(self, selected_algo: str):
        if selected_algo == "":
            return

        try:
            # Update the parameters panel
            schema = self.runner_widget.runner.get_parameters(selected_algo)

            self.params_panel.update(schema)

            self.napari_stack._on_layer_change(None)  # Refresh dropdowns in new UI

            # Update the number of samples available
            n_samples_available = self.runner_widget.runner.get_n_samples(selected_algo)
            self.runner_widget.update_n_samples(n_samples_available)

            # Check if tiled inference should be displayed or not
            algo_is_tileable = self.runner_widget.runner.is_tileable(selected_algo)
            self.runner_widget.update_tiled_ui(algo_is_tileable)

        except (AlgorithmServerError, ServerRequestError) as e:
            show_warning(e.message)

    def _run(self):
        algorithm_name = self.runner_widget.selected_algorithm_name
        if algorithm_name == "":
            return

        # Wrap the whole task preparation into a try/except block.
        try:
            algo_params = self.params_panel.get_algo_params()

            algo_param_defs = self.runner_widget.runner.get_parameters(algorithm_name)[
                "properties"
            ]

            signature_params = self.runner_widget.runner.get_signature_params(
                algorithm_name
            )

            resolved_params = etc.resolve_params(
                algo_param_defs,
                signature_params,
                args=(),
                algo_params=algo_params,
            )

            params_stack = Stack()
            for name, data in resolved_params.items():
                kw = algo_param_defs[name]
                kind = kw.pop("param_type")
                if "anyOf" in kw:
                    kw.pop("anyOf")  # added by Pydantic - we don't need it.
                param_layer = layer_factory(kind=kind, data=data, name=name, **kw)
                params_stack.add(param_layer)

            tiling_ctx = self.runner_widget.selected_tiling_specs

            task = partial(
                self.runner_widget.runner.run_generator,
                algorithm=algorithm_name,
                tiling_ctx=tiling_ctx,
                params_stack=params_stack,
            )

            self.reinitialize_domain = params_stack.extent

        except (AlgorithmServerError, ServerRequestError) as e:
            show_warning(e.message)
            return

        # Launch the task
        self.tasks.add_active(task, self._merge_wrap)

    def _merge_wrap(self, payload):
        if payload is None:
            return

        result_tile, params_domain = payload

        # New position can be offset by the position of the parameters tile
        if (result_tile.position is not None) and (params_domain.position is not None):
            result_tile.position = tuple(
                [p + q for p, q in zip(params_domain.position, result_tile.position)]
            )
        else:
            result_tile.position = params_domain.position

        if self.reinitialize_domain is None:
            # If inputs don't have an extent, we clear up the whole output
            self.reinitialize_domain = self.napari_stack.extent

        self.napari_stack.merge(result_tile, self.reinitialize_domain)

    def _download_sample(self, *args, **kwargs) -> Stack:
        algorithm_name = self.runner_widget.selected_algorithm_name
        if algorithm_name == "":
            return Stack()

        try:
            sample = self.runner_widget.runner.get_sample(
                algorithm_name, *args, **kwargs
            )
            if sample is not None:
                return sample
        except:
            show_warning("❌ Failed to download sample.")

        return Stack()

    def _sample_triggered(self):
        idx = self.runner_widget.samples_select.currentText()
        if idx == "":
            return

        self.tasks.add_active(
            task=partial(self._download_sample, idx=int(idx)),
            return_func=self._sample_emitted,
        )

    def _sample_emitted(self, sample: Stack):
        for sp in sample:
            if sp.kind in NAPARI_LAYER_TYPES:
                if sp.data is not None:
                    layer = layer_factory(
                        kind=sp.kind, name=sp.name, data=sp.data, meta=sp.meta
                    )
                    self.napari_stack.add(layer)
            else:
                # Set values in the parameters UI
                qt_widget_setter_func = self.params_panel.ui_state[
                    sp.name
                ].qt_widget_setter_func
                if qt_widget_setter_func is not None:
                    qt_widget_setter_func(sp.data)

    def _open_info_link_from_btn(self, *args, **kwargs) -> None:
        algorithm = self.runner_widget.selected_algorithm_name
        if algorithm == "":
            return

        self.runner_widget.runner.info(algorithm)

    def _connect_from_btn(self):
        server_url = self.runner_widget.server_url_field.text()
        if server_url == "":
            show_warning("Please specify a server URL!")
            return

        if isinstance(self.runner_widget.runner, Client):
            try:
                self.runner_widget.runner.connect(server_url)
            except (ServerRequestError, AlgorithmServerError) as e:
                show_warning(f"Could not connect to server on {server_url}")

        self.runner_widget.cb_algorithms.clear()
        self.runner_widget.cb_algorithms.addItems(self.runner_widget.runner.algorithms)

    def _cancel(self):
        show_info("Cancelling...")
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

