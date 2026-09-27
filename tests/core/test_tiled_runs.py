"""Tiled runs must give the same results as untiled runs.

Covers the tiling + merging pipeline end to end, for images (incl. channel layouts),
semantic and instance masks, and object layers (points, vectors, boxes).
"""

from pathlib import Path

import numpy as np
import pytest
import skimage.io
import skimage.measure

import imaging_server_kit as sk

SAMPLE_IMAGES_DIR = Path(__file__).parent.parent / "sample_images"


def _global_coords(layer: sk.Layer) -> np.ndarray:
    """Object coordinates (points, or vector origins) in global coordinates."""
    data = np.asarray(layer.data, dtype=float)
    position = np.asarray(layer.position, dtype=float)
    if layer.kind == "vectors":
        data = data.copy()
        data[:, 0] += position
        return data
    return data + position


def _sorted_rows(arr: np.ndarray) -> np.ndarray:
    """Sort objects (first axis) so that sets of objects can be compared regardless of order."""
    flat = np.asarray(arr).reshape(len(arr), -1)
    return flat[np.lexsort(flat.T[::-1])]


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two label images are identical up to renaming the labels."""
    if (a.shape != b.shape) or not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a[a > 0], b[b > 0]]), axis=1)  # Unique (label_a, label_b) pairs
    n_pairs = pairs.shape[1]
    return len(np.unique(pairs[0])) == n_pairs == len(np.unique(pairs[1]))


### 1. Images: tiled == untiled ###


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def pointwise(image: np.ndarray) -> sk.Image:
    return sk.Image(image * 2 + 1)


def _assert_tiled_image_matches_untiled(**tiling):
    image = np.random.default_rng(0).random((50, 70))
    expected = pointwise.run(image)[0]

    result = pointwise.run(image, tiled=True, **tiling)[0]

    assert result.data.shape == image.shape
    assert tuple(result.position) == (0, 0)
    np.testing.assert_allclose(result.data, expected.data, rtol=1e-6)


@pytest.mark.parametrize("tile_size", [16, 25, 100, [20, 30]])
@pytest.mark.parametrize("tile_overlap", [0.0, 0.25])
@pytest.mark.parametrize("tile_randomize", [False, True])
def test_tiled_image_matches_untiled(tile_size, tile_overlap, tile_randomize):
    _assert_tiled_image_matches_untiled(
        tile_size=tile_size, tile_overlap=tile_overlap, tile_randomize=tile_randomize
    )


@sk.algorithm(parameters={"image": sk.Image(channel_axis=0)}, tileable=True)
def pointwise_channel_first(image: np.ndarray) -> sk.Image:
    return sk.Image(image * 2 + 1, channel_axis=0)


@sk.algorithm(parameters={"image": sk.Image(channel_axis=2)}, tileable=True)
def pointwise_channel_last(image: np.ndarray) -> sk.Image:
    return sk.Image(image * 2 + 1, channel_axis=2)


@sk.algorithm(parameters={"image": sk.Image(rgb=True)}, tileable=True)
def pointwise_rgb(image: np.ndarray) -> sk.Image:
    return sk.Image(image * 2 + 1, rgb=True)


@pytest.mark.parametrize(
    "algo, shape, tile_size",
    [
        (pointwise_channel_first, (3, 50, 40), 16),
        (pointwise_channel_last, (50, 40, 3), 16),
        (pointwise_rgb, (50, 40, 3), 16),
        (pointwise, (10, 30, 40), [5, 16, 16]),  # 3D volume, 3D tiles
    ],
    ids=["channel_axis=0", "channel_axis=2", "rgb", "3d"],
)
@pytest.mark.parametrize("tile_overlap", [0.0, 0.25])
def test_tiled_channel_layouts_match_untiled(algo, shape, tile_size, tile_overlap):
    image = np.random.default_rng(1).random(shape)
    expected = algo.run(image)[0]

    result = algo.run(image, tiled=True, tile_size=tile_size, tile_overlap=tile_overlap)[0]

    assert result.data.shape == image.shape
    np.testing.assert_allclose(result.data, expected.data, rtol=1e-6)


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def threshold(image: np.ndarray) -> sk.Mask:
    return sk.Mask((image > 0.5).astype(np.int32) + (image > 0.8))


@pytest.mark.parametrize("tile_overlap", [0.0, 0.25])
def test_tiled_semantic_mask_matches_untiled(tile_overlap):
    image = np.random.default_rng(2).random((50, 70))
    expected = threshold.run(image)[0]

    result = threshold.run(image, tiled=True, tile_size=16, tile_overlap=tile_overlap)[0]

    np.testing.assert_array_equal(result.data, expected.data)


### 2. Instance masks ###


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def label_blobs(image: np.ndarray) -> sk.Mask:
    return sk.Mask(skimage.measure.label(image > 128), merger="instances")


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def label_every_pixel(image: np.ndarray) -> sk.Mask:
    labels = np.arange(1, image.size + 1).reshape(image.shape)
    return sk.Mask(labels, merger="instances")


@pytest.fixture(scope="module")
def blobs() -> np.ndarray:
    return skimage.io.imread(SAMPLE_IMAGES_DIR / "blobs.tif").astype(float)


@pytest.mark.parametrize("tile_size, tile_overlap", [(64, 0.25), (100, 0.1)])
def test_tiled_instance_segmentation_matches_untiled(blobs, tile_size, tile_overlap):
    """Objects crossing tile borders are stitched, and distinct objects are never merged."""
    expected = label_blobs.run(blobs)[0].data

    result = label_blobs.run(
        blobs, tiled=True, tile_size=tile_size, tile_overlap=tile_overlap
    )[0].data

    assert _same_partition(result, expected)


def test_tiled_instance_labels_from_first_tile_are_kept():
    """Labels of the first tile must not collide with those of the next tiles."""
    result = label_every_pixel.run(np.zeros((8, 8)), tiled=True, tile_size=4)[0].data

    assert len(np.unique(result)) == 64


def test_tiled_instance_mask_with_more_than_65535_labels():
    result = label_every_pixel.run(np.zeros((300, 300)), tiled=True, tile_size=128)[0].data

    assert result.dtype == np.uint32
    assert len(np.unique(result)) == 300 * 300


def test_tiled_instance_rerun_into_existing_stack(blobs):
    stack = label_blobs.run(blobs, tiled=True, tile_size=64, tile_overlap=0.25)
    stack = label_blobs.run(blobs, tiled=True, tile_size=64, tile_overlap=0.25, stack=stack)

    assert len(stack) == 1
    assert _same_partition(stack[0].data, label_blobs.run(blobs)[0].data)


### 3. Object layers ###


@sk.algorithm(parameters={"points": sk.Points()})
def count_points(points: np.ndarray) -> int:
    return len(points)


def test_points_only_input_receives_all_points():
    """The point at the maximum coordinate lies on the upper edge of the extent."""
    points = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])

    assert count_points(points) == 3


@sk.algorithm(parameters={"points": sk.Points()}, tileable=True)
def identity_points(points: np.ndarray) -> sk.Points:
    return sk.Points(points)


@sk.algorithm(parameters={"vectors": sk.Vectors()}, tileable=True)
def identity_vectors(vectors: np.ndarray) -> sk.Vectors:
    return sk.Vectors(vectors)


@sk.algorithm(parameters={"boxes": sk.Boxes()}, tileable=True)
def identity_boxes(boxes: np.ndarray) -> sk.Boxes:
    return sk.Boxes(boxes)


def _points_on_borders() -> np.ndarray:
    """Random points, plus points exactly on tile borders (tile size 32) and on the upper edge."""
    rng = np.random.default_rng(3)
    random_points = rng.integers(0, 100, (100, 2)).astype(float)
    edge_points = np.array([[32.0, 32.0], [64.0, 0.0], [0.0, 96.0], [99.0, 99.0]])
    return np.vstack([random_points, edge_points])


def test_tiled_points_are_returned_exactly_once():
    points = _points_on_borders()

    result = identity_points.run(points, tiled=True, tile_size=32)[0]

    np.testing.assert_array_equal(
        _sorted_rows(_global_coords(result)), _sorted_rows(points)
    )


def test_tiled_vectors_are_returned_exactly_once():
    origins = _points_on_borders()
    displacements = np.random.default_rng(4).random(origins.shape) * 10
    vectors = np.stack([origins, displacements], axis=1)

    result = identity_vectors.run(vectors, tiled=True, tile_size=32)[0]

    np.testing.assert_allclose(
        _sorted_rows(_global_coords(result)), _sorted_rows(vectors)
    )


def test_tiled_boxes_are_returned_exactly_once():
    """Boxes are assigned to the tile containing their center, even when they cross tile borders."""
    corners = _points_on_borders()
    boxes = np.stack(
        [corners, corners + [0, 6], corners + [6, 6], corners + [6, 0]], axis=1
    )

    result = identity_boxes.run(boxes, tiled=True, tile_size=32)[0]
    result_boxes = np.asarray(result.data) + np.asarray(result.position)

    np.testing.assert_allclose(_sorted_rows(result_boxes), _sorted_rows(boxes))


@sk.algorithm(parameters={"image": sk.Image()}, tileable=True)
def detect_pixels(image: np.ndarray) -> sk.Points:
    return sk.Points(np.argwhere(image > 0).astype(float))


def test_tiled_detection_matches_untiled():
    image = np.zeros((100, 100))
    rng = np.random.default_rng(5)
    image[rng.integers(0, 100, 60), rng.integers(0, 100, 60)] = 1
    image[[31, 32, 63, 64, 99], [31, 32, 63, 64, 99]] = 1  # On both sides of tile borders

    expected = detect_pixels.run(image)[0]
    result = detect_pixels.run(image, tiled=True, tile_size=32)[0]

    np.testing.assert_array_equal(
        _sorted_rows(_global_coords(result)), _sorted_rows(_global_coords(expected))
    )
