from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Optional
from imaging_server_kit.types import Layer


class Validator(ABC):
    """
    Methods
    -------
    validate():
        Validate a layer's internal attributes.
    """

    @staticmethod
    @abstractmethod
    def validate(layer: Optional[Layer]) -> None: ...


class DefaultValidator(Validator):
    @staticmethod
    def validate(layer: Optional[Layer]) -> None:
        pass
