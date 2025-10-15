"""
Serialization module for the Imaging Server Kit.
"""

from typing import Dict, List
import base64
import numpy as np
from imaging_server_kit.core.encoding import (
    encode_contents,
    decode_contents,
)
from imaging_server_kit.types import DATA_TYPES
from imaging_server_kit.core.results import Results


def _is_base64_encoded(data: str) -> bool:
    """
    Check if a given string is Base64-encoded.

    :param data: The string to check.
    :return: True if the string is Base64-encoded, otherwise False.
    """
    if not isinstance(data, str) or len(data) % 4 != 0:
        # Base64 strings must be divisible by 4
        return False

    try:
        # Try decoding and check if it re-encodes to the same value
        decoded_data = base64.b64decode(data, validate=True)
        return base64.b64encode(decoded_data).decode("utf-8") == data
    except Exception:
        return False


def _decode_data_features(data_params):
    """
    Decodes the `features` key of the data parameters.
    The `features` key represents measurements associated with labels, points, vectors, or tracks.
    """
    encoded_features = data_params.get("features")
    if encoded_features is not None:
        decoded_features = {}
        for key, val in encoded_features.items():
            if isinstance(val, str) and _is_base64_encoded(val):
                decoded_features[key] = decode_contents(val)
            else:
                decoded_features[key] = val
        data_params["features"] = decoded_features
    return data_params


def _encode_data_features(data_params):
    """
    Encodes the `features` key of the data parameters.
    The `features` key represents measurements associated with labels, points, vectors, or tracks.
    """
    features = data_params.get("features")
    if features is not None:
        # For these data types, we can pass features as numpy array but they must be encoded
        encoded_features = {
            key: (encode_contents(val) if isinstance(val, np.ndarray) else val)
            for (key, val) in features.items()
        }
        data_params["features"] = encoded_features
    return data_params


def serialize_results(results: Results) -> List[Dict]:
    """Serialize a Results object to JSON."""
    serialized_results = []
    for layer in results:
        serialized_results.append(
            {
                "kind": layer.kind,
                "data": type(layer).to_features(layer.data),
                "name": layer.name,
                "meta": _encode_data_features(layer.meta),
            }
        )
    return serialized_results


def deserialize_results(serialized_results: List[Dict]) -> Results:
    """Deserialize a JSON to a Results object."""
    results = Results()
    for result_dict in serialized_results:
        results.create(
            kind=result_dict.get("kind"),
            data=DATA_TYPES.get(result_dict.get("kind")).to_data(
                result_dict.get("data")
            ),
            name=result_dict.get("name"),
            meta=_decode_data_features(result_dict.get("meta")),
        )
    return results
