from typing import Dict, List, Tuple
import numpy as np
import imaging_server_kit as serverkit
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
                decoded_features[key] = serverkit.decode_contents(val)
            else:
                decoded_features[key] = val
        data_params['features'] = decoded_features
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
                serverkit.encode_contents(val)
                if isinstance(val, np.ndarray)
                else val
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
            features = serverkit.encode_contents(data.astype(np.float32))
        elif data_type == "labels":
            data = data.astype(np.uint16)
            features = serverkit.mask2features(data)
            data_params['image_shape'] = data.shape
        elif data_type == "labels3d":
            features = serverkit.encode_contents(data.astype(np.uint16))
        elif data_type == "points":
            features = serverkit.points2features(data)
        elif data_type == "points3d":
            features = serverkit.encode_contents(data.astype(np.float32))
        elif data_type == "boxes":
            features = serverkit.boxes2features(data)
            data_params['shape_type'] = 'rectangle'
        elif data_type == "vectors":
            features = serverkit.vectors2features(data)
        elif data_type == "tracks":
            features = serverkit.encode_contents(data.astype(np.float32))
        else:
            print(f"Unknown data_type: {data_type}")
        
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
            data = serverkit.decode_contents(features).astype(float)
        elif data_type == "labels":
            image_shape = data_params.pop('image_shape')
            data = serverkit.features2mask(features, image_shape)
        elif data_type == "labels3d":
            data = serverkit.decode_contents(features).astype(int)
        elif data_type == "points":
            data = serverkit.features2points(features)
        elif data_type == "points3d":
            data = serverkit.decode_contents(features).astype(float)
        elif data_type == "boxes":
            data = serverkit.features2boxes(features)
        elif data_type == "vectors":
            data = serverkit.features2vectors(features)
        elif data_type == "tracks":
            data = serverkit.decode_contents(features).astype(float)
        else:
            print(f"Unknown data_type: {data_type}")

        data_tuple = (data, data_params, data_type)

        result_data_tuple.append(data_tuple)

    return result_data_tuple
