"""Tests of the QuPath panel that do not need a QuPath connection."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("napari")
pytest.importorskip("qubalab")

from qtpy.QtWidgets import QApplication

import imaging_server_kit as sk
from imaging_server_kit.gui.qupath_serverkit.qupath_widget import QuPathWidget


@sk.algorithm(parameters={"image": sk.Image(), "threshold": sk.Float(default=0.5)})
def threshold(image: np.ndarray, threshold: float) -> sk.Mask:
    return sk.Mask(image > threshold)


@sk.algorithm(parameters={"image1": sk.Image(), "image2": sk.Image()})
def add_images(image1: np.ndarray, image2: np.ndarray) -> sk.Image:
    return sk.Image(image1 + image2)


@pytest.fixture(scope="module")
def qapp():
    # Qt widgets need an application (keep a reference, so it is not garbage-collected)
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(qapp):
    return QuPathWidget(runner=sk.combine([threshold, add_images]))


def test_only_lists_qupath_compatible_algorithms(widget):
    cb = widget.runner_widget.cb_algorithms
    assert [cb.itemText(i) for i in range(cb.count())] == ["threshold"]


def test_panel_hides_the_image_parameter(widget):
    # The image comes from QuPath
    assert list(widget.params_panel.ui_state) == ["threshold"]


def test_run_without_annotation_is_refused(widget):
    widget._run()

    assert "annotation is required" in widget.logger.toPlainText()


def test_connect_without_server_url(widget):
    widget._connect_from_btn()

    assert "Please specify a server URL!" in widget.logger.toPlainText()
