import numpy as np
import pytest
import skimage.data

import imaging_server_kit as sk
from imaging_server_kit.demo import examples


ALGORITHMS = [
    obj for obj in vars(examples).values() if isinstance(obj, sk.Algorithm)
]

IMAGE_ALGORITHMS = [
    examples.threshold_algo,
    examples.auto_threshold,
    examples.gaussian_algo,
    examples.sobel_algo,
    examples.blob_detector_algo,
    examples.nl_means_denoise,
    examples.slic_algo,
    examples.background_subtract,
    examples.project,
]


def _sample_ids(algos):
    return [
        (algo, idx) for algo in algos for idx in range(algo.get_n_samples())
    ]


## All examples ##


def test_all_examples_are_listed():
    assert set(ALGORITHMS) == set(IMAGE_ALGORITHMS) | {
        examples.fibonacci_sphere,
        examples.notif_stream,
        examples.conway_algo,
    }


@pytest.mark.parametrize("algo", ALGORITHMS, ids=lambda a: a.name)
def test_has_description(algo):
    assert algo.algo_info["description"]


@pytest.mark.parametrize(
    "algo, idx",
    _sample_ids(ALGORITHMS),
    ids=lambda x: x.name if isinstance(x, sk.Algorithm) else str(x),
)
def test_samples_run(algo, idx):
    sample = algo.get_sample(idx=idx)
    stack = algo.run(**{layer.name: layer.data for layer in sample.layers})
    assert all(layer.kind != "notification" for layer in stack)


@pytest.mark.parametrize(
    "algo",
    IMAGE_ALGORITHMS,
    ids=lambda a: a.name,
)
def test_missing_image_notifies(algo):
    stack = algo.run(image=None)
    assert stack[0].kind == "notification"


## Segmentation ##


def test_threshold():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert examples.threshold_algo(img, threshold=0.5, dark_background=True).sum() == 50
    assert examples.threshold_algo(img, threshold=0.5, dark_background=False).sum() == 50


def test_threshold_with_offset():
    img = np.arange(100, 200, dtype=np.uint8).reshape(10, 10)
    assert examples.threshold_algo(img, threshold=0.5, dark_background=True).sum() == 50


@pytest.mark.parametrize("method", ["Otsu", "Li"])
def test_auto_threshold(method):
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5:] = 200
    mask = examples.auto_threshold(img, method=method)
    np.testing.assert_array_equal(mask, img > 0)


def test_slic():
    mask = examples.slic_algo(skimage.data.astronaut()[:64, :64], n_segments=10)
    assert mask.shape == (64, 64)
    assert len(np.unique(mask)) > 1


## Filtering ##


@pytest.mark.parametrize(
    "algo",
    [
        examples.gaussian_algo,
        examples.sobel_algo,
        examples.nl_means_denoise,
        examples.background_subtract,
    ],
    ids=lambda a: a.name,
)
def test_filters_preserve_shape(algo):
    img = skimage.data.camera()[:64, :64]
    assert algo(img).shape == img.shape


@pytest.mark.parametrize("method, expected", [("subtract", 0.0), ("divide", 1.0)])
def test_background_subtract_constant_image(method, expected):
    img = np.full((32, 32), 50.0)
    out = examples.background_subtract(img, sigma=5.0, method=method)
    np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("method", ["max", "min", "mean", "std", "median"])
def test_project(method):
    img = np.random.default_rng(0).random((5, 16, 16))
    out = examples.project(img, method=method)
    np.testing.assert_allclose(out[0], getattr(np, method)(img, axis=0))


def test_project_tiled_matches_full():
    img = skimage.data.brain()
    full = examples.project(img, method="max")
    tiled = examples.project.run(image=img, method="max", tiled=True, tile_size=64)[0].data
    np.testing.assert_array_equal(tiled, full)


## Detection ##


def _blob_image(blob_value, background_value):
    img = np.full((64, 64), background_value, dtype=np.uint8)
    img[30:35, 30:35] = blob_value
    return img


def _detect(image, **kwargs):
    params = dict(min_sigma=1, max_sigma=5, num_sigma=5, threshold=0.1)
    return examples.blob_detector_algo.run(image=image, **{**params, **kwargs})


def test_blob_detector_bright_blob():
    stack = _detect(_blob_image(200, 0), invert_image=False, time_dim=False)
    points = stack[0].data
    assert len(points) == 1
    np.testing.assert_allclose(points[0], [32, 32], atol=1)
    assert stack[1].data == "Points detected: 1"


@pytest.mark.parametrize(
    "blob_value",
    [20, 0],
)
def test_blob_detector_dark_blob(blob_value):
    stack = _detect(_blob_image(blob_value, 200), invert_image=True, time_dim=False)
    assert stack[0].kind == "points"
    assert len(stack[0].data) == 1


def test_blob_detector_frame_by_frame():
    frame = _blob_image(200, 0)
    img = np.stack([frame, np.roll(frame, 10, axis=1)])
    points = _detect(img, invert_image=False, time_dim=True)[0].data
    np.testing.assert_allclose(points, [[0, 32, 32], [1, 32, 42]], atol=1)


def test_blob_detector_no_blob_notifies():
    stack = _detect(np.zeros((32, 32)), invert_image=False, time_dim=False)
    assert stack[0].kind == "notification"


## Others ##


def test_fibonacci_sphere():
    points = examples.fibonacci_sphere(N=50, r=5.0)
    assert points.shape == (50, 3)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 5.0)


def test_notifications_stream():
    stack = examples.notif_stream.run(time_delay=1, n_times=1, level="warning")
    assert stack[0].kind == "notification"
    assert stack[0].data == "Step: 0"


def test_game_of_life():
    stack = examples.conway_algo.run(max_iter=3, delay=0)
    mask = stack[0].data
    assert mask.shape == (100, 100)
    assert set(np.unique(mask)) <= {0, 255}


def test_game_of_life_progress_completes():
    stack = examples.conway_algo.run(max_iter=3, delay=0)
    assert stack[1].data == 3
