import re

import numpy as np
import pytest

import imaging_server_kit as sk


@pytest.mark.parametrize("cls, label", [(sk.Image, "Image"), (sk.Mask, "Mask")])
def test_array_layers_reject_empty_arrays(cls, label):
    with pytest.raises(ValueError, match=re.escape(f"{label} array has an invalid shape: (0, 3)")):
        cls(np.zeros((0, 3)))


@pytest.mark.parametrize("cls", [sk.Image, sk.Mask])
def test_array_layers_reject_non_arrays(cls):
    with pytest.raises(TypeError, match="is not a Numpy array"):
        cls([[1, 2], [3, 4]])


@pytest.mark.parametrize("cls", [sk.Image, sk.Mask])
def test_array_layers_check_dimensionality(cls):
    with pytest.raises(ValueError, match="wrong dimensionality"):
        cls(np.zeros((4, 4)), dimensionality=[3])


def test_rgb_image_needs_three_channels():
    with pytest.raises(ValueError, match="should be RGB"):
        sk.Image(np.zeros((4, 4, 2)), rgb=True)
