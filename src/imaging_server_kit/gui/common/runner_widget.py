from typing import Callable, Optional, List

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from napari_toolkit.containers.collapsible_groupbox import QCollapsibleGroupBox

from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.tiling import TilingSpecs


class RunnerWidget:
    def __init__(self, runner: AlgorithmRunner, algorithms: Optional[List[str]] = None):
        self.runner = runner

        # We can specify a subset of algorithms to display in the dropdown
        # (for example, when using to_qupath() we only display qupath-compatible algorithms)
        if algorithms is None:
            available_algorithms = self.runner.algorithms
        else:
            available_algorithms = [
                a for a in algorithms if a in self.runner.algorithms
            ]

        # Layout and widget
        self._widget = QWidget()
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self._widget.setLayout(layout)

        # Algorithms
        self.cb_algorithms = QComboBox()
        self.cb_algorithms.addItems(available_algorithms)
        layout.addWidget(QLabel("Algorithm"), 1, 0)
        layout.addWidget(self.cb_algorithms, 1, 1)

        # Info link
        self.algo_info_btn = QPushButton("🌐 Doc")
        self.algo_info_btn.clicked.connect(self._open_info_link_from_btn)
        layout.addWidget(self.algo_info_btn, 1, 2)

        # Samples
        self.samples_select = QComboBox()
        self.samples_select_btn = QPushButton("Load")
        self.samples_select_label = QLabel("Samples (0)")
        layout.addWidget(self.samples_select_label, 2, 0)
        layout.addWidget(self.samples_select, 2, 1)
        layout.addWidget(self.samples_select_btn, 2, 2)
        self.samples_select.setVisible(False)
        self.samples_select_btn.setVisible(False)
        self.samples_select_label.setVisible(False)

        # (Experimental) run in tiles
        self.experimental_gb = QCollapsibleGroupBox("Tiled inference")  # type: ignore
        self.experimental_gb.setChecked(False)
        experimental_layout = QGridLayout(self.experimental_gb)
        layout.addWidget(self.experimental_gb, 3, 0, 1, 3)

        experimental_layout.addWidget(QLabel("Run in tiles"), 0, 0)
        self.cb_run_in_tiles = QCheckBox()
        self.cb_run_in_tiles.setChecked(False)
        self.cb_run_in_tiles.toggled.connect(self._run_in_tiles_changed)
        experimental_layout.addWidget(self.cb_run_in_tiles, 0, 1)

        experimental_layout.addWidget(QLabel("Tile size [px]"), 1, 0)
        self.qds_tile_size = QSpinBox()
        self.qds_tile_size.setMinimum(16)
        self.qds_tile_size.setMaximum(4096)
        self.qds_tile_size.setSingleStep(16)
        self.qds_tile_size.setValue(128)
        self.qds_tile_size.setEnabled(False)
        experimental_layout.addWidget(self.qds_tile_size, 1, 1)

        experimental_layout.addWidget(QLabel("Overlap [0-1]"), 2, 0)
        self.qds_overlap = QDoubleSpinBox()
        self.qds_overlap.setMinimum(0)
        self.qds_overlap.setMaximum(1)
        self.qds_overlap.setSingleStep(0.01)
        self.qds_overlap.setValue(0)
        self.qds_overlap.setEnabled(False)
        experimental_layout.addWidget(self.qds_overlap, 2, 1)

        experimental_layout.addWidget(QLabel("Delay [sec]"), 3, 0)
        self.qds_delay = QDoubleSpinBox()
        self.qds_delay.setMinimum(0)
        self.qds_delay.setMaximum(1)
        self.qds_delay.setSingleStep(0.1)
        self.qds_delay.setValue(0)
        self.qds_delay.setEnabled(False)
        experimental_layout.addWidget(self.qds_delay, 3, 1)

        experimental_layout.addWidget(QLabel("Randomize"), 4, 0)
        self.cb_randomize = QCheckBox()
        self.cb_randomize.setChecked(False)
        self.cb_randomize.setEnabled(False)
        experimental_layout.addWidget(self.cb_randomize, 4, 1)

    @property
    def widget(self) -> QWidget:
        return self._widget

    @property
    def update_params_trigger(self) -> Callable:
        return self.cb_algorithms.currentTextChanged  # type: ignore

    @property
    def selected_algorithm_name(self) -> str:
        return self.cb_algorithms.currentText()

    @property
    def selected_tiling_specs(self) -> Optional[TilingSpecs]:
        tiled = self.cb_run_in_tiles.isChecked()
        if tiled:
            return TilingSpecs(
                tile_size=self.qds_tile_size.value(),
                tile_overlap=self.qds_overlap.value(),
                tile_delay=self.qds_delay.value(),
                tile_randomize=self.cb_randomize.isChecked(),
            )

    def update_n_samples(self, n_samples_available: int = 0):
        self.samples_select.clear()
        if n_samples_available == 0:
            self.samples_select.setVisible(False)
            self.samples_select_btn.setVisible(False)
            self.samples_select_label.setVisible(False)
        else:
            self.samples_select.setVisible(True)
            self.samples_select_btn.setVisible(True)
            self.samples_select_label.setVisible(True)
            self.samples_select.addItems([f"{k}" for k in range(n_samples_available)])
            self.samples_select_label.setText(f"Samples ({n_samples_available})")

    def update_tiled_ui(self, algo_is_tileable: bool = False):
        if algo_is_tileable is False:
            # Make sure NOT to run in tiled mode
            self.cb_run_in_tiles.setChecked(False)

        self.experimental_gb.setVisible(algo_is_tileable)

    def _open_info_link_from_btn(self, *args, **kwargs) -> None:
        algorithm: str = self.cb_algorithms.currentText()
        if algorithm == "":
            print("Seletcting an algorithm is required!")
            return

        self.runner.info(algorithm)

    def _run_in_tiles_changed(self, run_in_tiles: bool):
        for ui_element in [
            self.qds_tile_size,
            self.qds_overlap,
            self.qds_delay,
            self.cb_randomize,
        ]:
            ui_element.setEnabled(run_in_tiles)
