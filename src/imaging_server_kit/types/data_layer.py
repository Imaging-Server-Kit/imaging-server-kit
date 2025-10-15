from abc import ABC, abstractmethod
from typing import Dict, Optional, Union
import numpy as np


class DataLayer(ABC):
    def __init__(
        self,
        data=None,
        name: str = "",
        description: str = "",
        meta: Optional[Dict] = None,
    ):
        self.name = name
        self.description = description
        self.type = Union[str, np.ndarray]
        self.kind = None
        self.meta = meta if meta is not None else {}
        self.data = data
        if self.data is not None:
            self.validate_data(data, self.meta)

    def __str__(self) -> str:
        return f"DataLayer: {self.name} ({self.kind})"

    def _decode_and_validate(self, cls, v, meta):
        data = self.to_data(v)
        self.validate_data(data, meta)
        return data

    def update(self, updated_data: np.ndarray) -> np.ndarray:
        self.data = updated_data
        self.refresh()

    def refresh(self):
        pass

    @classmethod
    @abstractmethod
    def to_features(cls, data: np.ndarray): ...

    @classmethod
    @abstractmethod
    def to_data(cls, features): ...

    @classmethod
    def _merge_tile(
        cls, current_data: np.ndarray, tile_data: np.ndarray, tile_info: dict
    ) -> np.ndarray:
        return tile_data

    @classmethod
    def _get_tile(cls, current_data: np.ndarray, tile_info: dict) -> np.ndarray:
        return current_data

    @classmethod
    def _get_initial_data(cls, pixel_domain):
        pass

    @classmethod
    def pixel_domain(cls, data: np.ndarray):
        pass

    @classmethod
    def validate_data(cls, data, meta):
        pass
