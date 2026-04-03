from abc import ABC, abstractmethod
from typing import Any, Optional
from imaging_server_kit.types import Layer


class Serializer(ABC):
    """
    Methods
    -------
    serialize():
        Serializes the class into a JSON-compatible representation.
    deserialize():
        Reconstructs an instance from a JSON representation.
    """

    @staticmethod
    @abstractmethod
    def serialize(layer: Optional[Layer], client_origin: str) -> Any: ...

    @staticmethod
    @abstractmethod
    def deserialize(serialized_data: Any, client_origin: str) -> Any: ...


class DefaultDataSerializer(Serializer):
    @staticmethod
    def serialize(layer: Optional[Layer], client_origin: str) -> Any:
        if layer is not None:
            return layer.data

    @staticmethod
    def deserialize(serialized_data: Any, client_origin: str) -> Any:
        return serialized_data
