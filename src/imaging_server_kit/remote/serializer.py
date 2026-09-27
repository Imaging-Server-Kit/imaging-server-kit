from abc import ABC, abstractmethod
from typing import Any, Optional
from imaging_server_kit.types import Layer

# Key identifying the (last) message streamed by the server when an algorithm fails,
# e.g. {"__error__": {"type": "RuntimeError", "message": "..."}}
ERROR_FRAME_KEY = "__error__"


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
