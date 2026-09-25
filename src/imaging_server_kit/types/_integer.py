from typing import Optional
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core._fmt import fmt_num


class Integer(Layer):
    """Data layer used to represent integer values.

    Parameters
    ----------
    data: An integer value.
    min: Minimum accepted value.
    max: Maximum accepted value.
    step: Step size used by interactive sliders/spinboxes.
    default: Default value used when `data` is not provided.
    """

    kind = "int"
    type = Optional[int]

    def __init__(
        self,
        data: Optional[int] = None,
        name="Int",
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
            data=data,
            default=default,
            required=required,
            auto_call=auto_call,
            min=min,
            max=max,
            step=step,
            **kwargs,
        )

    def _summary(self) -> str:
        summary = super()._summary()
        value_range = (self.meta.get("min"), self.meta.get("max"))
        if value_range != (int(np.iinfo(np.int16).min), int(np.iinfo(np.int16).max)):
            summary += f" in [{fmt_num(value_range[0])}, {fmt_num(value_range[1])}]"
        return summary
