
from typing import Optional

from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types._null import Null


class NullDataSerializer(Serializer):
    @staticmethod
    def serialize(null: Optional[Null], client_origin: str) -> None:
        if null is None:
            return
        if null.data is not None:
            raise ValueError(f"Cannot serialize this object: {null.data}")
        return None

    @staticmethod
    def deserialize(serialized_data, client_origin: str) -> None:
        return None