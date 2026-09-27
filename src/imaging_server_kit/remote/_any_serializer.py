import typing
from typing import Optional

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._any import Any

from imaging_server_kit.remote.encoding import (
    decode_array,
    encode_array,
    is_encoded_array,
)


def _serialize_value(value: typing.Any) -> typing.Any:
    # Common types: None, str, int, float, bool are returned directly
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Numpy scalars are converted to Python scalars
    if isinstance(value, np.generic):
        return value.item()

    # Numpy arrays are encoded (tagged, to be told apart from strings)
    if isinstance(value, np.ndarray):
        return encode_array(value)

    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    # If unsuccessful with any of the previous approaches, we raise:
    raise ValueError(f"Cannot serialize this object: {value}")


def _deserialize_value(value: typing.Any) -> typing.Any:
    if is_encoded_array(value):
        return decode_array(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {key: _deserialize_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]

    raise ValueError(f"Cannot deserialize this object: {value}")


class AnyDataSerializer(Serializer):
    @staticmethod
    def serialize(any: Optional[Any]) -> typing.Any:
        if any is None:
            return None

        return _serialize_value(any.data)

    @staticmethod
    def deserialize(serialized_data: typing.Any) -> typing.Any:
        return _deserialize_value(serialized_data)
