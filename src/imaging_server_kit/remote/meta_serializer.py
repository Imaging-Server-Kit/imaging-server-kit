from typing import Any, Dict, Optional

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.remote.encoding import decode_value, encode_value
from imaging_server_kit.types.layer import Layer


class MetaSerializer(Serializer):
    @staticmethod
    def serialize(layer: Optional[Layer]) -> Optional[Dict]:
        if layer is not None:
            if layer.meta is not None:
                return encode_value(layer.meta)
            else:
                return {}

    @staticmethod
    def deserialize(serialized_meta: Dict) -> Any:
        return decode_value(serialized_meta)
