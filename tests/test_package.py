"""Tests of the top-level helpers of the package (`sk.run`, `sk.convert`)."""

import numpy as np
import pytest

import imaging_server_kit as sk


@sk.algorithm(parameters={"image": sk.Image(), "offset": sk.Integer(default=1)})
def add_offset(image: np.ndarray, offset: int) -> sk.Image:
    return sk.Image(image + offset, name="Result")


def test_run_is_equivalent_to_runner_run():
    image = np.zeros((4, 4))

    result = sk.run(add_offset, image, offset=3)

    assert isinstance(result, sk.Stack)
    np.testing.assert_array_equal(result.read("Result").data, add_offset.run(image, offset=3)[0].data)


def test_run_in_tiles():
    tileable = sk.Algorithm(lambda image: sk.Image(image * 2, name="Result"), tileable=True)
    image = np.arange(64.0).reshape(8, 8)

    result = sk.run(tileable, image, tiled=True, tile_size=4)

    np.testing.assert_allclose(result.read("Result").data, image * 2)


def test_convert_to_stack_copies_the_layer_list():
    stack = sk.Stack([sk.Image(np.zeros((4, 4)), name="a")])

    converted = sk.convert(stack)

    assert converted is not stack
    assert [l.name for l in converted] == ["a"]
    converted.delete("a")
    assert len(stack) == 1


def test_convert_to_unsupported_target_raises():
    with pytest.raises(ValueError, match="not supported"):
        sk.convert(sk.Stack(), to="zarr")
