from abc import ABC, abstractmethod
from typing import Any

class DataSerializer(ABC):
    @abstractmethod
    def serialize(self, data: Any, client_origin: str) -> Any: ...

    @abstractmethod
    def deserialize(self, serialized_data: Any, client_origin: str) -> Any: ...


class DefaultDataSerializer(DataSerializer):
    def serialize(self, data: Any, client_origin: str) -> Any:
        return data

    def deserialize(self, serialized_data: Any, client_origin: str) -> Any:
        return serialized_data