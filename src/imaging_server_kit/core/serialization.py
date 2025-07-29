from typing import Dict, List, Tuple
import numpy as np
from imaging_server_kit.core.encoding import (
    encode_contents,
    decode_contents,
)
from imaging_server_kit.core.geometry import (
    mask2features,
    instance_mask2features,
    points2features,
    boxes2features,
    vectors2features,
    features2mask,
    features2instance_mask,
    features2points,
    features2boxes,
    features2vectors,
)
import base64


def is_base64_encoded(data: str) -> bool:
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


def decode_data_features(data_params):
    """
    Decodes the `features` key of the data parameters.
    The `features` key represents measurements associated with labels, points, vectors, or tracks.
    """
    encoded_features = data_params.get("features")
    if encoded_features is not None:
        decoded_features = {}
        for key, val in encoded_features.items():
            if isinstance(val, str) and is_base64_encoded(val):
                decoded_features[key] = decode_contents(val)
            else:
                decoded_features[key] = val
        data_params["features"] = decoded_features
    return data_params


def encode_data_features(data_params):
    """
    Encodes the `features` key of the data parameters.
    The `features` key represents measurements associated with labels, points, vectors, or tracks.
    """
    features = data_params.get("features")
    if features is not None:
        # For these data types, we can pass features as numpy array but they must be encoded
        encoded_features = {
            key: (
                encode_contents(val) if isinstance(val, np.ndarray) else val
            )
            for (key, val) in features.items()
        }
        data_params["features"] = encoded_features
    return data_params


def serialize_result_tuple(result_data_tuple: List[Tuple]) -> List[Dict]:
    """Converts the result data tuple to dict that can be serialized as JSON (used by the server)."""
    serialized_results = []
    for data, data_params, data_type in result_data_tuple:
        data_params = encode_data_features(data_params)

        if data_type == "image":
            features = encode_contents(data.astype(np.float32))
        elif data_type == "mask":
            features = mask2features(data.astype(np.uint16))
            data_params["image_shape"] = data.shape
        elif data_type == "instance_mask":
            features = instance_mask2features(data.astype(np.uint16))
            data_params["image_shape"] = data.shape
        elif data_type == "mask3d":
            features = encode_contents(data.astype(np.uint16))
        elif data_type == "points":
            features = points2features(data)
        elif data_type == "points3d":
            features = encode_contents(data.astype(np.float32))
        elif data_type == "boxes":
            features = boxes2features(data)
            data_params["shape_type"] = "rectangle"
        elif data_type == "vectors":
            features = vectors2features(data)
        elif data_type == "tracks":
            features = encode_contents(data.astype(np.float32))
        elif data_type == "class":
            features = data  # A simple string
        elif data_type == "text":
            features = data  # A text string
        elif data_type == "notification":
            features = data
        elif data_type == "scalar":
            features = data
        elif data_type == "list":
            features = data  # Lists of numeric/string values don't need to be encoded
        else:
            print(f"Unknown data_type: {data_type}")
            features = None

        serialized_results.append(
            {
                "type": data_type,
                "data": features,
                "data_params": data_params,
            }
        )

    return serialized_results


def deserialize_result_tuple(serialized_results: List[Dict]) -> List[Tuple]:
    """Converts serialized JSON results to a results data tuple (used by the client)."""
    result_data_tuple = []
    for result_dict in serialized_results:
        data_type = result_dict.get("type")
        features = result_dict.get("data")
        data_params = result_dict.get("data_params")

        data_params = decode_data_features(data_params)

        if data_type == "image":
            data = decode_contents(features).astype(float)
        elif data_type == "mask":
            image_shape = data_params.pop("image_shape")
            data = features2mask(features, image_shape)
        elif data_type == "instance_mask":
            image_shape = data_params.pop("image_shape")
            data = features2instance_mask(features, image_shape)
        elif data_type == "mask3d":
            data = decode_contents(features).astype(int)
        elif data_type == "points":
            data = features2points(features)
        elif data_type == "points3d":
            data = decode_contents(features).astype(float)
        elif data_type == "boxes":
            data = features2boxes(features)
        elif data_type == "vectors":
            data = features2vectors(features)
        elif data_type == "tracks":
            data = decode_contents(features).astype(float)
        elif data_type == "class":
            data = features  # A simple string
        elif data_type == "text":
            data = features  # A text string
        elif data_type == "notification":
            data = features
        elif data_type == "scalar":
            data = features
        elif data_type == "list":
            data = features
        else:
            print(f"Unknown data_type: {data_type}")
            data = features

        data_tuple = (data, data_params, data_type)

        result_data_tuple.append(data_tuple)

    return result_data_tuple
