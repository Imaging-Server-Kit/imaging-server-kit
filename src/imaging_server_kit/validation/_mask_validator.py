
from typing import Optional

from imaging_server_kit.validation.validator import Validator
from imaging_server_kit.validation._array_validator import validate_array
from imaging_server_kit.types._mask import Mask


class MaskValidator(Validator):
    @staticmethod
    def validate(mask: Optional[Mask]) -> None:
        if mask is None:
            return

        validate_array(mask, "Mask")
