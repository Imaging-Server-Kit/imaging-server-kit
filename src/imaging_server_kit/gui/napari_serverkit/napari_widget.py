from functools import partial

import napari
from napari.utils.notifications import show_info, show_warning

import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.errors import AlgorithmServerError, ServerRequestError
from imaging_server_kit.core.stack import Stack
from imaging_server_kit.gui.common import NAPARI_LAYER_TYPES
from imaging_server_kit.gui.common.base_widget import BaseRunnerWidget
from imaging_server_kit.gui.napari_serverkit.napari_stack import NapariStack
from imaging_server_kit.types import layer_factory
from imaging_server_kit.core.runner import AlgorithmRunner


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
