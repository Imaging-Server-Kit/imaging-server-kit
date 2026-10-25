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
