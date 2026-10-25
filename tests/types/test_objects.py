import numpy as np
import pytest

import imaging_server_kit as sk


@pytest.mark.parametrize(
    "cls, data_2d, data_3d",
    [
        (sk.Points, np.zeros((1, 2)), np.zeros((1, 3))),
        (sk.Vectors, np.zeros((1, 2, 2)), np.zeros((1, 2, 3))),
        (sk.Boxes, np.zeros((1, 4, 2)), np.zeros((1, 4, 3))),
    ],
)
def test_dimensionality_is_enforced(cls, data_2d, data_3d):
    cls(data_2d, dimensionality=[2])
    with pytest.raises(ValueError):
        cls(data_3d, dimensionality=[2])

    # Any dimensionality is accepted by default
    cls(data_3d)


@sk.algorithm(parameters={"points": sk.Points(dimensionality=[2])})
def count_2d_points(points: np.ndarray) -> int:
    return len(points)


def test_points_parameter_dimensionality():
    assert count_2d_points(np.zeros((3, 2))) == 3
    with pytest.raises(ValueError):
        count_2d_points(np.zeros((3, 3)))


### Paths ###


def _two_paths() -> sk.Paths:
    return sk.Paths(
        [np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([[30.0, 30.0], [40.0, 40.0]])],
        features={"length": np.array([7.0, 14.0])},
    )


def test_paths_extent():
    paths = _two_paths()

    assert paths.coords_min == (0.0, 0.0)
    assert paths.coords_max == (41.0, 41.0)  # Exclusive upper bound, like Points


def test_paths_select_keeps_paths_centered_in_domain():
    selection = _two_paths().select(sk.Domain(position=(20, 20), size=(30, 30)))

    assert selection.n_objects == 1
    # Local to the domain, and unchanged in global coordinates
    np.testing.assert_array_equal(selection.data[0], [[10, 10], [20, 20]])
    assert tuple(selection.position) == (20, 20)
    np.testing.assert_array_equal(selection.meta["features"]["length"], [14.0])


def test_paths_select_on_positioned_layer():
    paths = sk.Paths([np.array([[0.0, 0.0], [2.0, 2.0]])], position=(10, 10))

    selection = paths.select(sk.Domain(position=(8, 8), size=(8, 8)))

    np.testing.assert_array_equal(np.asarray(selection.data[0]) + selection.position, [[10, 10], [12, 12]])


def test_paths_select_undefined_domain_keeps_everything():
    paths = _two_paths()

    selection = paths.select(sk.Domain())

    assert selection.n_objects == 2


### Tracks ###


def _tracks() -> sk.Tracks:
    # Columns: [ID, T, Y, X]
    return sk.Tracks(np.array([[1, 0, 10, 12], [1, 1, 30, 40], [2, 0, 5, 5]], dtype=float))


def test_tracks_extent():
    tracks = _tracks()

    assert tracks.coords_min == (0.0, 5.0, 5.0)
    assert tracks.coords_max == (2.0, 31.0, 41.0)


def test_tracks_select_keeps_points_in_domain():
    selection = _tracks().select(sk.Domain(position=(0, 20, 20), size=(2, 20, 30)))

    # Only the second point of track 1; the ID is not translated
    np.testing.assert_array_equal(selection.data, [[1, 1, 10, 20]])
    assert tuple(selection.position) == (0, 20, 20)


@sk.algorithm(parameters={"tracks": sk.Tracks()}, tileable=True)
def track_points(tracks: np.ndarray) -> sk.Points:
    return sk.Points(tracks[:, 1:], name="Track points")


def test_tracks_tiled_run_receives_each_point_once():
    tracks = _tracks()

    result = track_points.run(tracks.data, tiled=True, tile_size=16)

    points = result.read("Track points")
    global_points = points.data + points.position
    sort = lambda arr: arr[np.lexsort(arr.T[::-1])]
    np.testing.assert_array_equal(sort(global_points), sort(tracks.data[:, 1:]))
