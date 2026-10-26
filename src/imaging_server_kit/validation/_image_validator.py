
from typing import Optional

from imaging_server_kit.validation.validator import Validator
from imaging_server_kit.validation._array_validator import validate_array
from imaging_server_kit.types._image import Image


class ImageValidator(Validator):
    @staticmethod
    def validate(image: Optional[Image]) -> None:
        if image is None:
            return

        validate_array(image, "Image")

        data = image.data
        meta = image.meta

        if meta:
            if meta["rgb"] is True:
                if len(data.shape) not in [3, 4]:
                    raise ValueError("Image should be RGB.")
                if data.shape[-1] != 3:
                    raise ValueError("Image should be RGB.")
