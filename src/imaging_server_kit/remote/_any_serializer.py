import typing
from typing import Optional

import numpy as np

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._any import Any

from imaging_server_kit.remote.encoding import decode_contents, encode_contents


class AnyDataSerializer(Serializer):
    @staticmethod
    def serialize(any: Optional[Any], client_origin: str) -> typing.Any:
        if any is None:
            return None

        if any.data is None:
            return None

        value = any.data

        # If any.data is a common type; int, str, float, bool, we return it directly
        if isinstance(value, (str, int, float, bool)):
            return value

        # Numpy arrays are encoded normally;
        elif isinstance(value, np.ndarray):
            return encode_contents(value)

        # Dictionary case:
        elif isinstance(value, dict):
            return {
                key: AnyDataSerializer.serialize(Any(data=item), client_origin)
                for key, item in value.items()
            }

        # List/Tuple case:
        elif isinstance(value, (list, tuple)):
            return [
                AnyDataSerializer.serialize(Any(data=item), client_origin)
                for item in value
            ]
        
        else:
            # If unsuccessful with any of the previous approaches, we raise:
            raise ValueError(f"Cannot serialize this object: {value}")

    @staticmethod
    def deserialize(serialized_data: typing.Any, client_origin: str) -> typing.Any:
        if isinstance(serialized_data, str):
            try:
                return decode_contents(serialized_data)
            except Exception:
                return serialized_data
        
        elif isinstance(serialized_data, (int, float, bool)):
            return serialized_data
        
        elif isinstance(serialized_data, list):
            return [
                AnyDataSerializer.deserialize(item, client_origin)
                for item in serialized_data
            ]
        
        elif isinstance(serialized_data, dict):
            return {
                key: AnyDataSerializer.deserialize(item, client_origin)
                for key, item in serialized_data.items()
            }
        
        else:
            raise ValueError(f"Cannot deserialize this object: {serialized_data}")
