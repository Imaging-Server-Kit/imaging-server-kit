from functools import partial

import napari
from napari.utils.notifications import show_info, show_warning

from imaging_server_kit.core.errors import AlgorithmServerError, ServerRequestError
from imaging_server_kit.core.stack import Stack
from imaging_server_kit.gui.common import NAPARI_LAYER_TYPES
from imaging_server_kit.gui.common.base_widget import BaseRunnerWidget
from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack
from imaging_server_kit.types import layer_factory
from imaging_server_kit.core.runner import AlgorithmRunner, merge_result_tile


class NapariWidget(BaseRunnerWidget):
    def __init__(self, runner: AlgorithmRunner, viewer: napari.Viewer):
        super().__init__(runner)

        # Shared with NapariStack here...
        self.napari_stack = NapariStack(
            viewer, pbar=self.pbar, params_panel=self.params_panel
        )

        # Connect the samples loading event
        self.runner_widget.samples_select_btn.clicked.connect(self._sample_triggered)

        self._select_first_algorithm()

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

            params_stack = self.runner_widget.runner._params_stack(
                algorithm_name, args=(), algo_params=algo_params
            )

            tiling_ctx = self.runner_widget.selected_tiling_specs

            task = partial(
                self.runner_widget.runner.run_generator,
                algorithm=algorithm_name,
                tiling_ctx=tiling_ctx,
                params_stack=params_stack,
            )

            # The parameters extent doesn't change during the run (computed once)
            self.params_extent = params_stack.extent

        except (AlgorithmServerError, ServerRequestError) as e:
            show_warning(e.message)
            return

        # Launch the task
        self.tasks.add_active(task, self._merge_wrap)

    def _merge_wrap(self, payload):
        if payload is None:
            return

        result_tile, params_tile = payload

        merge_result_tile(self.napari_stack, result_tile, params_tile, self.params_extent)

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
        except Exception:
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

    def _log(self, message: str, warning: bool = False) -> None:
        if warning:
            show_warning(message)
        else:
            show_info(message)
