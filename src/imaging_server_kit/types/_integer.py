from typing import Optional
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core._fmt import fmt_num


class Integer(Layer):
    """Data layer for integer values.

    Parameters
    ----------
    data : int, optional
        The value.
    name : str, default="Int"
        Name of the layer.
    min : int, optional
        Minimum accepted value.
    max : int, optional
        Maximum accepted value.
    step : int, default=1
        Step size of the spin box in user interfaces.
    default : int, default=0
        Default value, used when `data` is not provided.
    auto_call : bool, default=False
        Re-run the algorithm when the value changes in user interfaces.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).

    Examples
    --------
    >>> threshold = sk.Integer(name="Threshold", min=0, max=255, default=128)
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
