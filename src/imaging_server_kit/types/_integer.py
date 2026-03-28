from typing import Optional
import numpy as np

from imaging_server_kit.types.data_layer import DataLayer


class Integer(DataLayer):
    """Data layer used to represent integer values."""

    kind = "int"
    type = int

    def __init__(
        self,
        data: Optional[int] = None,
        name="Int",
        description="Numeric parameter (integer)",
        default: int = 0,
        required: bool = True,
        auto_call: bool = False,
        min: int = int(np.iinfo(np.int16).min),
        max: int = int(np.iinfo(np.int16).max),
        step: int = 1,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            default=default,
            required=required,
            auto_call=auto_call,
            min=min,
            max=max,
            step=step,
            **kwargs,
        )