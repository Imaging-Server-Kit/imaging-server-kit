from typing import Dict, Optional

from imaging_server_kit.core.domain import Domain
from imaging_server_kit.remote.serializer import Serializer
from imaging_server_kit.types.layer import Layer


class DomainSerializer(Serializer):
    @staticmethod
    def serialize(layer: Optional[Layer], client_origin: str) -> Optional[Dict]:
        if layer is not None:
            if layer.domain is not None:
                return layer.domain.serialize()

    @staticmethod
    def deserialize(serialized_data: Optional[Dict], client_origin: str) -> Domain:
        if serialized_data is not None:
            return Domain(**serialized_data)
        else:
            return Domain()
