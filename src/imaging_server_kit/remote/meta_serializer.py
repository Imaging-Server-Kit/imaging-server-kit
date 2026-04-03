from typing import Any, Dict, Optional
import base64

import numpy as np
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import encode_contents, decode_contents
from imaging_server_kit.types.layer import Layer


class DefalutMetaSerializer(Serializer):
    @staticmethod
    def serialize(
        layer: Optional[Layer], client_origin: Optional[str] = None
    ) -> Optional[Dict]:
        if layer is not None:
            if layer.meta is not None:
                return _serialize_meta(layer.meta)
            else:
                return {}

    @staticmethod
    def deserialize(serialized_meta: Dict, client_origin: Optional[str] = None) -> Any:
        return _deserialize_meta(serialized_meta)


def _serialize_value(obj: Any) -> Any:
    if isinstance(obj, Dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return encode_contents(obj)
    return obj


def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively serialize Numpy arrays in the meta dictionary."""
    return {k: _serialize_value(v) for k, v in meta.items()}


def _is_base64_encoded(data: str) -> bool:
    """Check if a given string is Base64-encoded."""
    if not isinstance(data, str) or len(data) % 4 != 0:
        # Base64 strings must be divisible by 4
        return False
    try:
        # Try decoding and check if it re-encodes to the same value
        decoded_data = base64.b64decode(data, validate=True)
        return base64.b64encode(decoded_data).decode("utf-8") == data
    except Exception:
        return False


def _deserialize_value(obj: Any) -> Any:
    if isinstance(obj, Dict):
        return {k: _deserialize_value(v) for k, v in obj.items()}
    if isinstance(obj, str) and _is_base64_encoded(obj):
        # TODO: This is a bit sketchy - we use a try/except on the decoding to figure out
        # if the values in meta correspond to numpy arrays (features, etc.)
        try:
            return decode_contents(obj)
        except:
            return obj
    return obj


def _deserialize_meta(serialized_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively deserialize Numpy arrays in the meta dictionary."""
    return {k: _deserialize_value(v) for k, v in serialized_meta.items()}
