import imaging_server_kit as sk
from skimage.data import coins, astronaut
import numpy as np


### Merging images ###

# Merge two RGB images
def test_merge_rgb():
    mrg_rgb = sk.merge_layers(
        [sk.Image(astronaut(), rgb=True), sk.Image(astronaut(), rgb=True, position=(20, 25))]
    )
    assert len(mrg_rgb.shape) == 3


# Merge two gray images
def test_merge_gray():
    mrg_gray = sk.merge_layers(
        [sk.Image(coins()), sk.Image(coins(), position=(20, 25))]
    )
    assert len(mrg_gray.shape) == 2


# Merge two channel images
def test_merge_channel_images():
    mrg_chan = sk.merge_layers(
        [
            sk.Image(np.random.random((4, 100, 120)), channel_axis=0),
            sk.Image(np.random.random((4, 140, 110)), channel_axis=0, position=(20, 25)),
        ]
    )
    assert len(mrg_chan.shape) == 3
