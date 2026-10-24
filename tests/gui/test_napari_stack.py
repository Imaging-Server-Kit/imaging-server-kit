import os

import numpy as np
import pytest
import skimage.measure

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
napari = pytest.importorskip("napari")

import imaging_server_kit as sk


@sk.algorithm(parameters={"image": sk.Image()})
def label_with_features(image: np.ndarray) -> sk.Mask:
    labels = skimage.measure.label(image > 0)
    features = skimage.measure.regionprops_table(labels, properties=("label", "area"))
    return sk.Mask(labels, merger="instances", features=features)


@pytest.fixture
def viewer():
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()


def test_rerun_in_domain_into_viewer(viewer):
    """Each `run(stack=viewer)` creates a new NapariStack: the merger and features of existing layers are recovered."""
    image = np.zeros((64, 64))
    image[4:8, 4:8] = image[4:8, 40:44] = image[40:44, 4:8] = 1

    label_with_features.run(image, stack=viewer)
    label_with_features.run(image, stack=viewer, domain=sk.Domain(position=(0, 0), size=(32, 32)))

    labels = viewer.layers[0].data
    assert len(np.unique(labels[labels > 0])) == 3  # No label collisions

    features = viewer.layers[0].features.dropna()
    assert len(features) == 3
    for label, area in zip(features["label"], features["area"]):
        assert (labels == label).sum() == area
