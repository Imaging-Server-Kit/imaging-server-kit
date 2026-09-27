import io
import base64
from typing import Any, Dict

import numpy as np
import tifffile


def decode_contents(contents: str) -> np.ndarray:
    """Decodes base64 encoded image contents into a NumPy array.

    Args:
        contents (str): The base64 encoded string of the image.

    Returns:
        np.ndarray: The decoded image represented as a NumPy array.
    """
    return tifffile.imread(io.BytesIO(base64.b64decode(contents)))


def encode_contents(arr: np.ndarray) -> str:
    """Encodes a NumPy array image into a base64 string.

    Args:
        image (np.ndarray): The image to encode.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    img_byte_arr = io.BytesIO()
    tifffile.imwrite(img_byte_arr, arr)

    return base64.b64encode(img_byte_arr.getvalue()).decode()


# Key identifying encoded Numpy arrays in meta dictionaries and `Any` data, e.g. {"__ndarray__": "<base64>"}
ARRAY_KEY = "__ndarray__"


def encode_array(arr: np.ndarray) -> Dict[str, str]:
    """Encode a Numpy array as a tagged dictionary, so that it can be told apart from plain strings."""
    return {ARRAY_KEY: encode_contents(arr)}


def is_encoded_array(obj: Any) -> bool:
    """Whether `obj` is a Numpy array encoded with `encode_array()`."""
    return isinstance(obj, dict) and (obj.keys() == {ARRAY_KEY})


def decode_array(obj: Dict[str, str]) -> np.ndarray:
    """Decode a Numpy array encoded with `encode_array()`."""
    return decode_contents(obj[ARRAY_KEY])
