"""Errors raised to the user: algorithm failures, misuse of tiling, invalid parameters reported by a server."""

import numpy as np
import pytest

import imaging_server_kit as sk
from imaging_server_kit.core.errors import (
    AlgorithmRuntimeError,
    InvalidAlgorithmParametersError,
)


@sk.algorithm
def fails(x: int = 1) -> int:
    raise RuntimeError("boom")


@sk.algorithm
def fails_after_yield(x: int = 1) -> int:
    yield x
    raise RuntimeError("boom")


@sk.algorithm(parameters={"image": sk.Image()})
def not_tileable(image: np.ndarray) -> sk.Image:
    return sk.Image(image)


@pytest.mark.parametrize("algo", [fails, fails_after_yield], ids=["plain", "generator"])
def test_algorithm_error_is_wrapped(algo):
    with pytest.raises(AlgorithmRuntimeError, match="boom") as exc_info:
        algo.run()

    assert isinstance(exc_info.value.error, RuntimeError)


def test_tiled_run_of_non_tileable_algorithm_raises():
    with pytest.raises(AlgorithmRuntimeError, match="tiled"):
        not_tileable.run(np.zeros((8, 8)), tiled=True)


@pytest.mark.parametrize(
    "response_body, expected_in_message",
    [
        (
            {"detail": [{"loc": ["offset"], "msg": "Input should be less than 10", "input": 999}]},
            ["offset", "Input should be less than 10", "999"],
        ),
        ({"detail": "Algorithm not found"}, ["Algorithm not found"]),
        ("Unprocessable entity", ["Unprocessable entity"]),
    ],
    ids=["pydantic-detail", "string-detail", "plain-text"],
)
def test_invalid_parameters_error_message(response_body, expected_in_message):
    error = InvalidAlgorithmParametersError(422, response_body)

    for expected in expected_in_message:
        assert expected in str(error)
