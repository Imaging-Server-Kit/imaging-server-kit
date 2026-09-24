import numpy as np
import pytest
from shapely.geometry import shape
from skimage.filters import gaussian
from skimage.measure import label

from imaging_server_kit.types._mask import (
    features2instance_mask,
    features2mask,
    instance_mask2features,
    mask2features,
)


@pytest.fixture
def instance_mask():
    m = np.zeros((12, 12), dtype=np.uint16)
    m[1:8, 1:8] = 1  # Donut
    m[3:5, 3:5] = 0
    m[9:11, 9:11] = 2  # Two blobs touching diagonally
    m[11, 11] = 2
    m[0, 11] = 3  # Single pixel
    return m


@pytest.fixture
def random_label_image():
    rng = np.random.default_rng(0)
    return label(gaussian(rng.random((256, 256)), 3) > 0.5).astype(np.int32)


def _features_by_id(features):
    return {f["properties"]["Detection ID"]: f for f in features}


def test_instance_one_feature_per_label(instance_mask):
    features = instance_mask2features(instance_mask)
    assert [f["properties"]["Detection ID"] for f in features] == [1, 2, 3]


def test_instance_hole_is_interior_ring(instance_mask):
    donut = shape(_features_by_id(instance_mask2features(instance_mask))[1]["geometry"])
    assert donut.geom_type == "Polygon"
    assert len(donut.interiors) == 1
    assert donut.area == (instance_mask == 1).sum()


def test_instance_disconnected_parts_are_multipolygon(instance_mask):
    geom = shape(_features_by_id(instance_mask2features(instance_mask))[2]["geometry"])
    assert geom.geom_type == "MultiPolygon"
    assert len(geom.geoms) == 2


def test_single_pixel_is_kept(instance_mask):
    geom = shape(_features_by_id(instance_mask2features(instance_mask))[3]["geometry"])
    assert geom.area == 1
    assert geom.bounds == (11, 0, 12, 1)  # Pixel edges, in (x, y)


def test_semantic_one_feature_per_component(instance_mask):
    features = mask2features(instance_mask)
    classes = sorted(f["properties"]["Class"] for f in features)
    assert classes == [1, 2, 2, 3]
    ids = [f["properties"]["Detection ID"] for f in features]
    assert len(set(ids)) == len(ids)


def test_instance_roundtrip(instance_mask, random_label_image):
    for m in (instance_mask, random_label_image):
        out = features2instance_mask(instance_mask2features(m), m.shape)
        np.testing.assert_array_equal(out, m)


def test_semantic_roundtrip():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[2:10, 2:10] = 1
    m[4:6, 4:6] = 2  # Class 2 inside a hole of class 1
    m[12:18, 3:15] = 3
    m[14:16, 5:7] = 0
    m[0, :] = 1
    out = features2mask(mask2features(m), m.shape)
    np.testing.assert_array_equal(out, m)


def test_bool_roundtrip():
    m = np.zeros((10, 10), dtype=bool)
    m[2:8, 2:8] = True
    m[4, 4] = False
    out = features2mask(mask2features(m), m.shape)
    np.testing.assert_array_equal(out.astype(bool), m)


def test_offset(instance_mask):
    offset = (100, 50)  # (row, col)
    features = instance_mask2features(instance_mask, offset=offset)
    geom = shape(_features_by_id(features)[3]["geometry"])
    assert geom.bounds == (61, 100, 62, 101)
    out = features2instance_mask(features, instance_mask.shape, offset=offset)
    np.testing.assert_array_equal(out, instance_mask)


def test_int64_labels(random_label_image):
    m = random_label_image.astype(np.int64)
    out = features2instance_mask(instance_mask2features(m), m.shape)
    np.testing.assert_array_equal(out, m)


def test_empty_mask():
    m = np.zeros((8, 8), dtype=np.uint16)
    assert mask2features(m) == []
    assert instance_mask2features(m) == []
    np.testing.assert_array_equal(features2mask([], m.shape), m)
    np.testing.assert_array_equal(features2instance_mask([], m.shape), m)


def test_3d_mask_raises():
    with pytest.raises(ValueError):
        instance_mask2features(np.zeros((2, 4, 4), dtype=np.uint16))
