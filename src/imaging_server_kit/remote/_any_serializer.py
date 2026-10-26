import typing
from typing import Optional

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_value, encode_value
from imaging_server_kit.types._any import Any


class AnyDataSerializer(Serializer):
    @staticmethod
    def serialize(any: Optional[Any]) -> typing.Any:
        if any is None:
            return None

        # Only JSON-compatible values (and Numpy arrays) can be serialized
        return encode_value(any.data, strict=True)

    @staticmethod
    def deserialize(serialized_data: typing.Any) -> typing.Any:
        return decode_value(serialized_data, strict=True)
