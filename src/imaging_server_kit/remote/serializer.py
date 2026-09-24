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
    def serialize(layer: Optional[Layer]) -> Any: ...

    @staticmethod
    @abstractmethod
    def deserialize(serialized_data: Any) -> Any: ...


class DefaultDataSerializer(Serializer):
    @staticmethod
    def serialize(layer: Optional[Layer]) -> Any:
        if layer is not None:
            return layer.data

    @staticmethod
    def deserialize(serialized_data: Any) -> Any:
        return serialized_data
