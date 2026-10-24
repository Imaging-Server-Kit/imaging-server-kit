import numpy as np
import pytest

import imaging_server_kit as sk
from imaging_server_kit.demo import tools


ALGORITHMS = [
    obj for obj in vars(tools).values() if isinstance(obj, sk.Algorithm)
]

IMAGE_ALGORITHMS = [
    tools.gaussian_algo,
    tools.median_algo,
    tools.variance_algo,
    tools.sobel_algo,
    tools.invert_algo,
    tools.rescale_intensity_algo,
    tools.blobness,
    tools.rescale_algo,
    tools.crop_algo,
    tools.blob_detector_algo,
    tools.threshold_manual,
    tools.otsu_threshold,
]

MASK_ALGORITHMS = [
    tools.label_algo,
    tools.remove_small_objects_algo,
    tools.keep_biggest_objects_algo,
    tools.fill_holes_algo,
    tools.invert_mask_algo,
    tools.morphological_operators,
]


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 32), dtype=np.uint8)


@pytest.fixture
def binary_mask():
    """Three square objects of 4, 16 and 36 pixels."""
    m = np.zeros((20, 20), dtype=np.uint8)
    m[1:3, 1:3] = 1
    m[5:9, 5:9] = 1
    m[12:18, 12:18] = 1
    return m


def _sample_ids(algos):
    return [
        (algo, idx) for algo in algos for idx in range(algo.get_n_samples())
    ]


## All tools ##


def test_all_tools_are_listed():
    assert set(ALGORITHMS) == set(IMAGE_ALGORITHMS + MASK_ALGORITHMS) | {
        tools.math_operators_images,
        tools.math_operators_algo,
        tools.logical_operators_masks,
        tools.image_masking_algo,
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
    assert len(stack) == 1
    assert stack[0].kind != "notification"


@pytest.mark.parametrize("algo", IMAGE_ALGORITHMS, ids=lambda a: a.name)
def test_missing_image_notifies(algo):
    stack = algo.run(image=None)
    assert stack[0].kind == "notification"


@pytest.mark.parametrize("algo", MASK_ALGORITHMS, ids=lambda a: a.name)
def test_missing_mask_notifies(algo):
    stack = algo.run(mask=None)
    assert stack[0].kind == "notification"


## Process ##


@pytest.mark.parametrize(
    "algo",
    [
        tools.gaussian_algo,
        tools.median_algo,
        tools.variance_algo,
        tools.sobel_algo,
        tools.blobness,
    ],
    ids=lambda a: a.name,
)
def test_filters_preserve_shape(algo, image):
    assert algo(image).shape == image.shape


def test_variance_of_constant_image_is_zero():
    out = tools.variance_algo(np.full((16, 16), 7, dtype=np.uint8), size=3)
    np.testing.assert_allclose(out, 0, atol=1e-9)


def test_median_3d():
    img = np.zeros((8, 16, 16), dtype=np.uint8)
    assert tools.median_algo(img, size=1).shape == img.shape


def test_gaussian_rgb_does_not_mix_channels():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[..., 0] = 255
    out = tools.gaussian_algo(img, sigma=2, rgb=True)
    np.testing.assert_allclose(out[..., 0], 255)
    np.testing.assert_allclose(out[..., 1:], 0)


def test_invert_uint8(image):
    out = tools.invert_algo(image)
    assert out.dtype == image.dtype
    np.testing.assert_array_equal(
        out.astype(int), int(image.max()) + int(image.min()) - image.astype(int)
    )


def test_invert_bool():
    img = np.array([[True, False]])
    np.testing.assert_array_equal(tools.invert_algo(img), [[False, True]])


def test_rescale_intensity(image):
    out = tools.rescale_intensity_algo(image, min_val=-1.0, max_val=1.0)
    assert out.min() == pytest.approx(-1.0)
    assert out.max() == pytest.approx(1.0)


def test_rescale_intensity_invalid_range_notifies(image):
    stack = tools.rescale_intensity_algo.run(image=image, min_val=1.0, max_val=0.0)
    assert stack[0].kind == "notification"


## Utils ##


def test_rescale(image):
    assert tools.rescale_algo(image, scale=0.5, order=1).shape == (16, 16)


def test_rescale_rgb():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    assert tools.rescale_algo(img, scale=0.5, rgb=True).shape == (16, 16, 3)


@pytest.mark.parametrize("algo", [tools.gaussian_algo, tools.rescale_algo], ids=lambda a: a.name)
@pytest.mark.parametrize("rgb", [True, False])
def test_rgb_output_layer(algo, rgb):
    img = np.zeros((32, 32, 3) if rgb else (32, 32), dtype=np.uint8)
    layer = algo.run(image=img, rgb=rgb)[0]
    assert layer.meta["rgb"] is rgb


def test_crop(image):
    assert tools.crop_algo(image, crop_width=4).shape == (24, 24)


## Detect ##


def test_blob_detector_finds_blob():
    img = np.zeros((64, 64))
    img[30:35, 30:35] = 1.0
    points = tools.blob_detector_algo(
        img, min_sigma=1, max_sigma=5, num_sigma=5, threshold=0.1
    )
    assert len(points) == 1
    np.testing.assert_allclose(points[0], [32, 32], atol=1)


def test_blob_detector_no_blob_notifies():
    stack = tools.blob_detector_algo.run(image=np.zeros((32, 32)))
    assert stack[0].kind == "notification"


## Segmentation ##


@pytest.mark.parametrize("dark_background", [True, False])
def test_threshold_manual_zero_based(dark_background):
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    mask = tools.threshold_manual(img, threshold=0.5, dark_background=dark_background)
    assert mask.sum() == 50


def test_threshold_manual_with_offset():
    img = np.arange(100, 200, dtype=np.uint8).reshape(10, 10)
    mask = tools.threshold_manual(img, threshold=0.5, dark_background=True)
    assert mask.sum() == 50


def test_otsu_threshold():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:, 5:] = 200
    mask = tools.otsu_threshold(img)
    np.testing.assert_array_equal(mask, img > 0)


## Mask ##


def test_label(binary_mask):
    assert tools.label_algo(binary_mask).max() == 3


@pytest.mark.parametrize("mask_is_binary", [True, False])
def test_remove_small_objects(binary_mask, mask_is_binary):
    from skimage.morphology import label

    mask = binary_mask if mask_is_binary else label(binary_mask)
    out = tools.remove_small_objects_algo(
        mask, min_size=10, mask_is_binary=mask_is_binary
    )
    assert (out > 0).sum() == 16 + 36


def test_remove_small_objects_keeps_objects_of_min_size(binary_mask):
    out = tools.remove_small_objects_algo(
        binary_mask, min_size=16, mask_is_binary=True
    )
    assert out.sum() == 16 + 36


def test_remove_small_objects_keeps_binary_dtype(binary_mask):
    out = tools.remove_small_objects_algo(
        binary_mask, min_size=10, mask_is_binary=True
    )
    assert out.dtype == binary_mask.dtype
    assert set(np.unique(out)) == {0, 1}


@pytest.mark.parametrize("n_objects, expected", [(1, 36), (2, 36 + 16), (5, 56)])
def test_keep_biggest_objects(binary_mask, n_objects, expected):
    out = tools.keep_biggest_objects_algo(
        binary_mask, n_objects=n_objects, mask_is_binary=True
    )
    assert (out > 0).sum() == expected


def test_keep_biggest_objects_returns_binary(binary_mask):
    out = tools.keep_biggest_objects_algo(binary_mask, n_objects=2, mask_is_binary=True)
    assert out.dtype == binary_mask.dtype
    assert set(np.unique(out)) == {0, 1}


def test_fill_holes():
    m = np.zeros((7, 7), dtype=np.uint8)
    m[1:6, 1:6] = 1
    m[3, 3] = 0
    assert tools.fill_holes_algo(m).sum() == 25


def test_invert_mask(binary_mask):
    out = tools.invert_mask_algo(binary_mask)
    assert out.dtype == binary_mask.dtype
    np.testing.assert_array_equal(out, 1 - binary_mask)


@pytest.mark.parametrize(
    "operation, expected",
    [("erosion", 1), ("dilation", 21), ("opening", 5), ("closing", 9)],
)
def test_morphological_operators(operation, expected):
    m = np.zeros((9, 9), dtype=bool)
    m[3:6, 3:6] = True
    out = tools.morphological_operators(m, operation=operation, radius=1)
    assert out.sum() == expected


## Math ##


@pytest.mark.parametrize(
    "operation, expected",
    [("add", 6.0), ("subtract", 2.0), ("multiply", 8.0), ("divide", 2.0)],
)
def test_math_image_to_image(operation, expected):
    a = np.full((4, 4), 4.0)
    b = np.full((4, 4), 2.0)
    out = tools.math_operators_images(a, b, operation=operation)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "operation, expected",
    [("add", 6.0), ("subtract", 2.0), ("multiply", 8.0), ("divide", 2.0)],
)
def test_math_image_with_value(operation, expected):
    out = tools.math_operators_algo(np.full((4, 4), 4.0), value=2.0, operation=operation)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    "operation, expected", [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])]
)
def test_math_mask_to_mask(operation, expected):
    m1 = np.array([[0, 0, 1, 1]], dtype=np.uint8)
    m2 = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    out = tools.logical_operators_masks(m1, m2, operation=operation)
    np.testing.assert_array_equal(out.astype(int), [expected])


def test_math_image_to_image_uint8_no_overflow():
    a = np.full((2, 2), 100, dtype=np.uint8)
    b = np.full((2, 2), 200, dtype=np.uint8)
    out = tools.math_operators_images(a, b, operation="subtract")
    np.testing.assert_allclose(out, -100)


def test_math_two_inputs_required(image, binary_mask):
    assert tools.math_operators_images.run(image1=image, image2=None)[0].kind == "notification"
    assert tools.logical_operators_masks.run(mask1=binary_mask, mask2=None)[0].kind == "notification"


def test_image_masking(image):
    mask = np.zeros_like(image)
    mask[:4, :4] = 1
    out = tools.image_masking_algo(image, mask, value=0)
    assert (out[:4, :4] == 0).all()
    np.testing.assert_array_equal(out[4:], image[4:])
    assert image[:4, :4].any()  # The input is left untouched


@pytest.mark.parametrize("rgb", [True, False])
def test_image_masking_rgb(rgb):
    img = np.full((8, 8, 3) if rgb else (8, 8), 255, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:4] = 1
    layer = tools.image_masking_algo.run(image=img, mask=mask, value=0, rgb=rgb)[0]
    assert layer.meta["rgb"] is rgb
    assert (layer.data[:4] == 0).all()
    assert (layer.data[4:] == 255).all()


## Tiling ##


def _tiling_cases():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (70, 90), dtype=np.uint8)
    img2 = rng.integers(0, 256, (70, 90), dtype=np.uint8)
    rgb = rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)
    m1 = (rng.random((70, 90)) > 0.5).astype(np.uint8)
    m2 = (rng.random((70, 90)) > 0.5).astype(np.uint8)
    return [
        (tools.invert_mask_algo, dict(mask=m1)),
        (tools.math_operators_images, dict(image1=img, image2=img2, operation="subtract")),
        (tools.math_operators_algo, dict(image=img, value=3.5, operation="divide")),
        (tools.logical_operators_masks, dict(mask1=m1, mask2=m2, operation="XOR")),
        (tools.image_masking_algo, dict(image=img, mask=m1, value=7, rgb=False)),
        (tools.image_masking_algo, dict(image=rgb, mask=m1, value=7, rgb=True)),
    ]


def test_tileable_tools_are_tested():
    # `label_algo` is tileable, but stitching objects across tiles is approximate
    tested = {algo for algo, _ in _tiling_cases()} | {tools.label_algo}
    assert {algo for algo in ALGORITHMS if algo.tileable} == tested


@pytest.mark.parametrize(
    "algo, params", _tiling_cases(), ids=lambda x: x.name if isinstance(x, sk.Algorithm) else ""
)
@pytest.mark.parametrize("tile_overlap", [0.0, 0.25, 0.6])
def test_tiled_matches_untiled(algo, params, tile_overlap):
    expected = algo.run(**params)[0].data
    result = algo.run(
        **params, tiled=True, tile_size=32, tile_overlap=tile_overlap, tile_randomize=True
    )[0].data
    assert result.shape == expected.shape
    np.testing.assert_allclose(result.astype(float), expected.astype(float), rtol=1e-6)
