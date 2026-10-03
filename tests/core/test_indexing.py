import numpy as np
import pytest

from imaging_server_kit.core.stack import Stack
from imaging_server_kit.types import Boxes, Image, Points, Vectors


@pytest.fixture
def arr():
    return np.arange(100).reshape(10, 10)


@pytest.fixture
def im(arr):
    return Image(arr, name="a")


@pytest.mark.parametrize(
    "key, expected",
    [
        (3, np.s_[3:4]),
        (-1, np.s_[-1:]),
        (np.s_[-3:], np.s_[-3:]),
        (np.s_[2:-2, 1:4], np.s_[2:-2, 1:4]),
        (np.s_[:, 2:5], np.s_[:, 2:5]),
        (np.s_[..., 2:5], np.s_[..., 2:5]),
        (np.s_[2:20], np.s_[2:20]),
        (np.s_[4, -2], np.s_[4:5, -2:-1]),
    ],
)
def test_layer_indexing(im, arr, key, expected):
    np.testing.assert_array_equal(im[key].data, arr[expected])


@pytest.mark.parametrize(
    "key, error",
    [
        (10, IndexError),
        (-11, IndexError),
        (np.s_[1, 2, 3], IndexError),
        (np.s_[..., ...], IndexError),
        (np.s_[::2], ValueError),
    ],
)
def test_layer_indexing_errors(im, key, error):
    with pytest.raises(error):
        im[key]


def test_layer_indexing_is_local(arr):
    im = Image(arr, name="a")
    im.position = (5, 5)
    sel = im[0:2]
    np.testing.assert_array_equal(sel.data, arr[0:2])
    assert tuple(sel.position) == (5, 5)


def test_stack_indexing(arr):
    stack = Stack([Image(arr, name="a"), Image(arr * 2, name="b")])

    cropped = stack[:, -3:]
    assert [l.name for l in cropped] == ["a", "b"]
    np.testing.assert_array_equal(cropped[0].data, arr[-3:])
    np.testing.assert_array_equal(cropped[1].data, arr[-3:] * 2)

    assert stack[0, 3].data.shape == (1, 10)
    np.testing.assert_array_equal(stack[1, ..., 2:5].data, arr[:, 2:5] * 2)
    assert [l.name for l in stack[::-1]] == ["b", "a"]


def test_stack_indexing_errors(arr):
    with pytest.raises(IndexError):
        Stack([Image(None, name="x")])[0, 1:3]
    with pytest.raises(IndexError):
        Stack([Image(arr, name="a")])[..., 1:3]


def _global(layer):
    """Layer data in global coordinates."""
    return (layer.data + np.asarray(layer.position)).tolist()


@pytest.fixture
def points():
    return Points(np.array([[2.0, 3.0], [4.0, 4.0], [6.0, 8.0]]), name="p")


@pytest.mark.parametrize(
    "key, expected",
    [
        (np.s_[4:], [[4.0, 4.0], [6.0, 8.0]]),
        (np.s_[0:3], [[2.0, 3.0]]),
        (4, [[4.0, 4.0]]),
        (np.s_[-1:], [[6.0, 8.0]]),
        (np.s_[:, 4:5], [[4.0, 4.0]]),
    ],
)
def test_points_indexing(points, key, expected):
    # Indices are counted from the layer's position, not from the data's bounding box
    assert _global(points[key]) == expected


def test_points_indexing_with_position():
    p = Points(np.array([[2.0, 3.0], [6.0, 8.0]]), name="p")
    p.position = (10, 10)
    assert _global(p[0:5]) == [[12.0, 13.0]]


def test_vectors_indexing():
    v = Vectors(
        np.array([[[2.0, 3.0], [1.0, 1.0]], [[6.0, 8.0], [1.0, 0.0]]]), name="v"
    )
    sel = v[4:]
    # Selected by origin; the direction is unchanged
    np.testing.assert_array_equal(sel.data[:, 0] + np.asarray(sel.position), [[6.0, 8.0]])
    np.testing.assert_array_equal(sel.data[:, 1], [[1.0, 0.0]])


def test_boxes_indexing():
    corners = np.array(
        [
            [[2, 2], [2, 4], [4, 4], [4, 2]],  # center (3, 3)
            [[4, 4], [4, 8], [8, 8], [8, 4]],  # center (6, 6)
        ],
        dtype=float,
    )
    b = Boxes(corners, name="b")
    # Selected by center, keeping all corners
    np.testing.assert_array_equal(np.asarray(_global(b[5:])), corners[1:])


def test_stack_indexing_object_layers(arr, points):
    alone = Stack([points])[0, 4:]
    mixed = Stack([Image(arr, name="a"), points])[1, 4:]
    assert _global(alone) == _global(mixed) == [[4.0, 4.0], [6.0, 8.0]]
