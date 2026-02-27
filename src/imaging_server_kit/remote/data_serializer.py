from abc import ABC, abstractmethod
from typing import Any, Optional
from imaging_server_kit.types import DataLayer


class Serializer(ABC):
    """
    Methods
    -------
    serialize():
        Serializes the class into a JSON-compatible representation.
    deserialize():
        Reconstructs an instance from a JSON representation.
    """
    @abstractmethod
    def serialize(self, layer: Optional[DataLayer], client_origin: str) -> Any: ...

    @abstractmethod
    def deserialize(self, serialized_data: Any, client_origin: str) -> Any: ...


class DefaultDataSerializer(Serializer):
    def serialize(self, layer: Optional[DataLayer], client_origin: str) -> Any:
        if layer is not None:
            return layer.data

    def deserialize(self, serialized_data: Any, client_origin: str) -> Any:
        return serialized_data
