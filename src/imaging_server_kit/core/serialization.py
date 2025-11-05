"""
Serialization module for the Imaging Server Kit.
"""

import base64
from typing import Any, Dict, List, Type
from imaging_server_kit.core.results import Results, DataLayer
from imaging_server_kit.types import DATA_TYPES

from imaging_server_kit.core.encoding import decode_contents


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


def serialize_results(results: Results, client_origin: str) -> List[Dict]:
    """Serialize a Results object to JSON."""
    return results.serialize(client_origin=client_origin)


def deserialize_results(serialized_results: List[Dict], client_origin: str) -> Results:
    """Deserialize a JSON to a Results object."""
    results = Results()
    
    for ser in serialized_results:
        kind = ser["kind"]
        name = ser["name"]
        meta = ser["meta"]
        data = ser["data"]

        cls: Type[DataLayer] = DATA_TYPES[kind]
        decoded_data = cls.deserialize(data, client_origin)
        decoded_meta = _deserialize_meta(meta)

        results.create(
            kind=kind,
            name=name,
            data=decoded_data,
            meta=decoded_meta,
        )
    
    return results
