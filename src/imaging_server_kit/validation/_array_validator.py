import numpy as np

from imaging_server_kit.types import Layer


def validate_array(layer: Layer, label: str) -> None:
    """Checks shared by array layers (Image, Mask): a non-empty Numpy array with an accepted dimensionality."""
    data = layer.data
    meta = layer.meta

    if not isinstance(data, np.ndarray):
        raise TypeError(f"{label} data ({type(data)}) is not a Numpy array")

    if not all(data.shape):
        raise ValueError(f"{label} array has an invalid shape: {data.shape}")

    if meta:
        if len(data.shape) not in meta["dimensionality"]:
            raise ValueError(f"{label} array has the wrong dimensionality.")
