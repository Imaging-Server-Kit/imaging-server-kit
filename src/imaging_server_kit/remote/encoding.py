import io
import base64
from typing import Any, Dict, Optional, Union

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


def encode_contents(arr: np.ndarray, photometric: Optional[str] = None) -> str:
    """Encodes a NumPy array image into a base64 string.

    Args:
        image (np.ndarray): The image to encode.
        photometric (str, optional): Passed to `tifffile.imwrite` (by default, tifffile guesses it from the shape).

    Returns:
        str: The base64 encoded string representation of the image.
    """
    img_byte_arr = io.BytesIO()
    tifffile.imwrite(img_byte_arr, arr, photometric=photometric)

    return base64.b64encode(img_byte_arr.getvalue()).decode()


# Key identifying encoded Numpy arrays in meta dictionaries and `Any` data, e.g. {"__ndarray__": "<base64>"}
ARRAY_KEY = "__ndarray__"

# Key holding the dtype of string and object arrays, which are encoded as lists, e.g. {"__ndarray__": ["a", "b"], "__dtype__": "<U1"}
DTYPE_KEY = "__dtype__"


def encode_array(arr: np.ndarray) -> Dict[str, Any]:
    """Encode a Numpy array as a tagged dictionary, so that it can be told apart from plain strings."""
    if arr.dtype.kind in "UO":
        # TIFF only holds numeric data: string and object arrays (e.g. class names) are sent as lists
        return {ARRAY_KEY: arr.tolist(), DTYPE_KEY: arr.dtype.str}
    return {ARRAY_KEY: encode_contents(arr)}


def is_encoded_array(obj: Any) -> bool:
    """Whether `obj` is a Numpy array encoded with `encode_array()`."""
    return (
        isinstance(obj, dict)
        and (ARRAY_KEY in obj)
        and (obj.keys() <= {ARRAY_KEY, DTYPE_KEY})
    )


def decode_array(obj: Dict[str, Any]) -> np.ndarray:
    """Decode a Numpy array encoded with `encode_array()`."""
    if DTYPE_KEY in obj:
        return np.array(obj[ARRAY_KEY], dtype=obj[DTYPE_KEY])
    return decode_contents(obj[ARRAY_KEY])


# Key holding the shape of empty coordinate arrays, which are not sent as TIFF, e.g. {"__shape__": [0, 4, 2]}
SHAPE_KEY = "__shape__"


def encode_coords(arr: np.ndarray) -> Union[str, Dict[str, Any]]:
    """Encode the coordinates of objects (Points, Boxes, Vectors, Tracks) as float32."""
    if arr.size == 0:
        # TIFF does not support zero-size arrays: we only send the shape
        return {SHAPE_KEY: list(arr.shape)}
    # `minisblack`: arrays such as (N, 4, 3) boxes must not be stored as RGB images
    return encode_contents(arr.astype(np.float32), photometric="minisblack")


def decode_coords(obj: Any) -> Optional[np.ndarray]:
    """Decode coordinates encoded with `encode_coords()` (None if `obj` holds no coordinates)."""
    if isinstance(obj, dict) and (SHAPE_KEY in obj):
        return np.zeros(obj[SHAPE_KEY], dtype=float)
    if isinstance(obj, str):
        return decode_contents(obj).astype(float)


def encode_value(obj: Any, strict: bool = False) -> Any:
    """Recursively encode Numpy arrays (tagged) and Numpy scalars (as Python scalars) in dicts, lists and tuples.

    With `strict=True`, objects that are not JSON-compatible raise a ValueError (otherwise, they are returned as-is).
    """
    if isinstance(obj, dict):
        return {k: encode_value(v, strict) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_value(v, strict) for v in obj]
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if strict and not (obj is None or isinstance(obj, (str, int, float, bool))):
        raise ValueError(f"Cannot serialize this object: {obj}")
    return obj


def decode_value(obj: Any, strict: bool = False) -> Any:
    """Recursively decode the Numpy arrays encoded by `encode_value()`.

    With `strict=True`, objects that are not JSON-compatible raise a ValueError (otherwise, they are returned as-is).
    """
    if is_encoded_array(obj):
        return decode_array(obj)
    if isinstance(obj, dict):
        return {k: decode_value(v, strict) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_value(v, strict) for v in obj]
    if strict and not (obj is None or isinstance(obj, (str, int, float, bool))):
        raise ValueError(f"Cannot deserialize this object: {obj}")
    return obj
