"""Tests of the Napari widget, driven headless: set parameters in the panel, run, and check the viewer layers."""

import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
napari = pytest.importorskip("napari")

from qtpy.QtWidgets import QApplication

import imaging_server_kit as sk
from imaging_server_kit.gui.napari_serverkit.napari_widget import NapariWidget

SAMPLE_IMAGE = np.random.default_rng(0).random((64, 64))


@sk.algorithm(
    parameters={
        "image": sk.Image(),
        "threshold": sk.Float(default=0.5, min=0.0, max=1.0),
    },
    samples=[{"image": SAMPLE_IMAGE, "threshold": 0.3}],
    tileable=True,
)
def threshold(image: np.ndarray, threshold: float) -> sk.Mask:
    return sk.Mask(image > threshold, name="Mask")


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def detect_bright_pixels(image: np.ndarray) -> sk.Points:
    return sk.Points(np.argwhere(image > 0.99).astype(float), name="Detections")


@sk.algorithm(parameters={"mask": sk.Mask()})
def invert(mask: np.ndarray) -> sk.Mask:
    return sk.Mask(mask == 0, name="Inverted")


# Shapes of the images received by `record_tiles` (to check that tiling is applied)
RECEIVED_SHAPES = []


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def record_tiles(image: np.ndarray) -> sk.Image:
    RECEIVED_SHAPES.append(image.shape)
    return sk.Image(image, name="Copy")


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def corner_point_with_offset(image: np.ndarray) -> sk.Points:
    """One point at the top-left corner of each tile, returned with its own position (offset)."""
    return sk.Points(np.zeros((1, 2)), position=(2, 3), name="Corners")


@sk.algorithm(parameters={"n": sk.Integer(default=3, min=1, max=10)})
def n_points(n: int) -> sk.Points:
    """An algorithm without spatial inputs."""
    return sk.Points(np.stack([np.arange(n), np.arange(n)], axis=1).astype(float), name="Line")


def _wait_for_tasks(widget: NapariWidget, timeout: float = 30.0) -> None:
    """Process Qt events until the widget's background workers have finished."""
    t0 = time.time()
    while widget.tasks.active_workers:
        if time.time() - t0 > timeout:
            raise TimeoutError("The widget task did not finish in time.")
        QApplication.processEvents()
        time.sleep(0.01)
    for _ in range(10):
        QApplication.processEvents()


def _run(widget: NapariWidget) -> None:
    widget._run()
    _wait_for_tasks(widget)


def _set_param(widget: NapariWidget, name: str, value) -> None:
    widget.params_panel.ui_state[name].qt_widget_setter_func(value)


def _select_layer(widget: NapariWidget, param: str, layer_name: str) -> None:
    widget.params_panel.ui_state[param].qt_widget.setCurrentText(layer_name)


def _set_tiled(widget: NapariWidget, tile_size: int, overlap: float = 0.0) -> None:
    runner_widget = widget.runner_widget
    runner_widget.cb_run_in_tiles.setChecked(True)
    runner_widget.qds_tile_size.setValue(tile_size)
    runner_widget.qds_overlap.setValue(overlap)


@pytest.fixture(autouse=True)
def qt_exceptions(monkeypatch):
    """Collect exceptions raised in Qt callbacks (otherwise, PyQt aborts the test session), and fail on them."""
    exceptions = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc_info: exceptions.append(exc_info))
    yield exceptions
    assert not exceptions, f"Exceptions raised in Qt callbacks: {[e[1] for e in exceptions]}"


@pytest.fixture
def viewer():
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


@pytest.fixture
def threshold_widget(viewer):
    viewer.add_image(SAMPLE_IMAGE, name="image")
    return NapariWidget(runner=threshold, viewer=viewer)


### Parameters panel ###


def test_panel_shows_parameters_with_defaults(threshold_widget):
    ui_state = threshold_widget.params_panel.ui_state

    assert list(ui_state) == ["image", "threshold"]
    assert ui_state["threshold"].qt_widget.value() == pytest.approx(0.5)


def test_layer_dropdowns_follow_viewer_layers(threshold_widget, viewer):
    combobox = threshold_widget.params_panel.ui_state["image"].qt_widget

    assert [combobox.itemText(i) for i in range(combobox.count())] == ["image"]

    viewer.add_image(np.zeros((8, 8)), name="other")
    viewer.add_labels(np.zeros((8, 8), dtype=np.uint8), name="labels")  # Not an image
    assert [combobox.itemText(i) for i in range(combobox.count())] == ["image", "other"]


def test_panel_follows_selected_algorithm(viewer):
    widget = NapariWidget(runner=sk.combine([threshold, invert]), viewer=viewer)
    assert list(widget.params_panel.ui_state) == ["image", "threshold"]

    widget.runner_widget.cb_algorithms.setCurrentText("invert")
    assert list(widget.params_panel.ui_state) == ["mask"]

    widget.runner_widget.cb_algorithms.setCurrentText("threshold")
    assert list(widget.params_panel.ui_state) == ["image", "threshold"]


def test_layer_dropdowns_are_replaced_on_algorithm_change(viewer):
    widget = NapariWidget(runner=sk.combine([threshold, invert]), viewer=viewer)

    widget.runner_widget.cb_algorithms.setCurrentText("invert")
    widget.runner_widget.cb_algorithms.setCurrentText("threshold")

    layer_comboboxes = widget.params_panel.layer_comboboxes
    assert list(layer_comboboxes) == ["image"]
    assert layer_comboboxes["image"] == [widget.params_panel.ui_state["image"].qt_widget]


def test_tiled_options_only_for_tileable_algorithms(viewer):
    widget = NapariWidget(runner=sk.combine([threshold, invert]), viewer=viewer)
    assert not widget.runner_widget.experimental_gb.isHidden()

    widget.runner_widget.cb_algorithms.setCurrentText("invert")
    assert widget.runner_widget.experimental_gb.isHidden()


### Running ###


def test_run_adds_result_layer(threshold_widget, viewer):
    _set_param(threshold_widget, "threshold", 0.7)

    _run(threshold_widget)

    np.testing.assert_array_equal(viewer.layers["Mask"].data, SAMPLE_IMAGE > 0.7)
    assert threshold_widget.run_btn.isEnabled()


def test_run_uses_selected_layer(threshold_widget, viewer):
    viewer.add_image(np.ones((10, 12)), name="ones")
    _select_layer(threshold_widget, "image", "ones")

    _run(threshold_widget)

    np.testing.assert_array_equal(viewer.layers["Mask"].data, np.ones((10, 12), dtype=bool))


def test_rerun_replaces_result_layer(threshold_widget, viewer):
    _run(threshold_widget)
    _set_param(threshold_widget, "threshold", 0.9)
    _run(threshold_widget)

    assert [l.name for l in viewer.layers] == ["image", "Mask"]
    np.testing.assert_array_equal(viewer.layers["Mask"].data, SAMPLE_IMAGE > 0.9)


@pytest.mark.parametrize("tile_size, overlap", [(16, 0.0), (32, 0.25)])
def test_tiled_run_matches_untiled(threshold_widget, viewer, tile_size, overlap):
    _set_tiled(threshold_widget, tile_size, overlap)

    _run(threshold_widget)

    np.testing.assert_array_equal(viewer.layers["Mask"].data, SAMPLE_IMAGE > 0.5)
    assert viewer.layers["Mask"].data.shape == SAMPLE_IMAGE.shape


def test_tiled_run_processes_tiles(viewer):
    viewer.add_image(SAMPLE_IMAGE, name="image")
    widget = NapariWidget(runner=record_tiles, viewer=viewer)
    _set_tiled(widget, tile_size=32)
    RECEIVED_SHAPES.clear()

    _run(widget)

    assert RECEIVED_SHAPES == [(32, 32)] * 4
    # Tiled images are merged as float32
    np.testing.assert_allclose(viewer.layers["Copy"].data, SAMPLE_IMAGE, rtol=1e-6)


def test_tiled_run_offsets_results_by_their_own_position(viewer):
    viewer.add_image(SAMPLE_IMAGE, name="image")
    widget = NapariWidget(runner=corner_point_with_offset, viewer=viewer)
    _set_tiled(widget, tile_size=32)

    _run(widget)

    layer = viewer.layers["Corners"]
    result = layer.data + layer.translate
    expected = np.array([[2, 3], [2, 35], [34, 3], [34, 35]], dtype=float)
    np.testing.assert_array_equal(result[np.lexsort(result.T[::-1])], expected)


def test_rerun_on_smaller_input_only_replaces_results_in_its_extent(viewer):
    """The previous results are only cleared in the extent of the new inputs."""
    viewer.add_image(SAMPLE_IMAGE, name="image")
    viewer.add_image(np.zeros((16, 16)), name="small")
    widget = NapariWidget(runner=detect_bright_pixels, viewer=viewer)
    _run(widget)
    _select_layer(widget, "image", "small")
    _run(widget)

    all_detections = np.argwhere(SAMPLE_IMAGE > 0.99)
    outside_small = all_detections[(all_detections >= 16).any(axis=1)]
    layer = viewer.layers["Detections"]
    result = layer.data + layer.translate
    sort = lambda arr: arr[np.lexsort(arr.T[::-1])]
    assert len(outside_small) < len(all_detections)  # Some detections are removed
    np.testing.assert_array_equal(sort(result), sort(outside_small.astype(float)))


def test_rerun_without_spatial_inputs_replaces_result_layer(viewer):
    widget = NapariWidget(runner=n_points, viewer=viewer)
    _run(widget)
    _set_param(widget, "n", 2)
    _run(widget)

    np.testing.assert_array_equal(viewer.layers["Line"].data, [[0, 0], [1, 1]])


def test_tiled_run_places_objects_in_global_coordinates(viewer):
    viewer.add_image(SAMPLE_IMAGE, name="image")
    widget = NapariWidget(runner=detect_bright_pixels, viewer=viewer)
    _set_tiled(widget, tile_size=16)

    _run(widget)

    expected = np.argwhere(SAMPLE_IMAGE > 0.99)
    result = viewer.layers["Detections"].data + viewer.layers["Detections"].translate
    sort = lambda arr: arr[np.lexsort(arr.T[::-1])]
    np.testing.assert_array_equal(sort(result), sort(expected.astype(float)))


def test_run_with_mask_input(viewer):
    viewer.add_labels(np.eye(6, dtype=np.uint8), name="labels")
    widget = NapariWidget(runner=invert, viewer=viewer)

    _run(widget)

    np.testing.assert_array_equal(viewer.layers["Inverted"].data, np.eye(6) == 0)


### Samples ###


def test_load_sample_adds_layers_and_sets_parameters(viewer):
    widget = NapariWidget(runner=threshold, viewer=viewer)
    assert widget.runner_widget.samples_select.count() == 1

    widget._sample_triggered()
    _wait_for_tasks(widget)

    np.testing.assert_array_equal(viewer.layers["image"].data, SAMPLE_IMAGE)
    assert widget.params_panel.ui_state["threshold"].qt_widget.value() == pytest.approx(0.3)

    # The sample can then be used to run the algorithm
    _run(widget)
    np.testing.assert_array_equal(viewer.layers["Mask"].data, SAMPLE_IMAGE > 0.3)


def test_spin_boxes_accept_zero_bounds(viewer):
    @sk.algorithm(
        parameters={
            "offset": sk.Float(min=-10.0, max=0.0, default=-1.0),
            "count": sk.Integer(min=-5, max=0, default=-1),
        }
    )
    def zero_bounds(offset, count):
        return offset + count

    widget = NapariWidget(runner=zero_bounds, viewer=viewer)
    ui_state = widget.params_panel.ui_state

    assert ui_state["offset"].qt_widget.maximum() == 0.0
    assert ui_state["count"].qt_widget.maximum() == 0
