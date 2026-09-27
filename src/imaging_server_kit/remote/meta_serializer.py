from typing import Any, Dict, Optional

import numpy as np
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import (
    decode_array,
    encode_array,
    is_encoded_array,
)
from imaging_server_kit.types.layer import Layer


class MetaSerializer(Serializer):
    @staticmethod
    def serialize(layer: Optional[Layer]) -> Optional[Dict]:
        if layer is not None:
            if layer.meta is not None:
                return _serialize_value(layer.meta)
            else:
                return {}

    @staticmethod
    def deserialize(serialized_meta: Dict) -> Any:
        return _deserialize_value(serialized_meta)


def _serialize_value(obj: Any) -> Any:
    """Recursively encode Numpy arrays (tagged) and Numpy scalars (as Python scalars)."""
    if isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return encode_array(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _deserialize_value(obj: Any) -> Any:
    """Recursively decode the Numpy arrays encoded by `_serialize_value()`."""
    if is_encoded_array(obj):
        return decode_array(obj)
    if isinstance(obj, dict):
        return {k: _deserialize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_value(v) for v in obj]
    return obj
