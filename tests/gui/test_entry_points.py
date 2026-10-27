"""Smoke tests of the GUI entry points: `sk.to_napari` and the widgets opened from the Napari plugin menu."""

import importlib
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
napari = pytest.importorskip("napari")

import imaging_server_kit as sk
from imaging_server_kit.gui.napari_serverkit.extra_widgets import (
    AlgorithmDemosWidget,
    AlgorithmToolsWidget,
)
from imaging_server_kit.gui.napari_serverkit.napari_http_widget import NapariHttpWidget
from qtpy.QtWidgets import QLabel


@pytest.fixture
def viewer():
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


def test_to_napari_docks_a_widget_for_a_function(viewer):
    def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        return image

    result = sk.to_napari(gaussian_blur, viewer=viewer)

    assert result is viewer
    assert "gaussian_blur" in viewer.window.dock_widgets


def test_to_qwidget_accepts_a_collection(viewer):
    widget = sk.to_qwidget(sk.combine([lambda image: image]), viewer=viewer)

    assert widget.runner_widget.cb_algorithms.count() == 1


@pytest.mark.parametrize("widget_cls", [AlgorithmToolsWidget, AlgorithmDemosWidget])
def test_plugin_menu_algorithm_widgets(viewer, widget_cls):
    widget = widget_cls(viewer)

    assert widget.layout().count() > 0


def test_plugin_menu_server_widget_without_server(viewer):
    widget = NapariHttpWidget(viewer)

    # No server connected yet: the URL field is shown, and no algorithm is listed
    assert not widget.runner_widget.server_url_field.isHidden()
    assert widget.runner_widget.cb_algorithms.count() == 0


def test_plugin_menu_qupath_widget_without_qubalab(viewer, monkeypatch):
    # Simulate an install without the `qupath` extra: Napari must still import the widget module
    monkeypatch.setitem(sys.modules, "qubalab", None)
    for name in ["qupath_http_widget", "qupath_widget", "qupath_bridge"]:
        monkeypatch.delitem(sys.modules, f"imaging_server_kit.gui.qupath_serverkit.{name}", raising=False)

    module = importlib.import_module("imaging_server_kit.gui.qupath_serverkit.qupath_http_widget")
    widget = module.QuPathHttpWidget(viewer)

    label = widget.findChild(QLabel)
    assert "imaging-server-kit[qupath]" in label.text()
