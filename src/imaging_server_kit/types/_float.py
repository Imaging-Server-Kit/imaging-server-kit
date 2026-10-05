from typing import Optional
import numpy as np

from imaging_server_kit.types.layer import Layer
from imaging_server_kit.core._fmt import fmt_num


class Float(Layer):
    """Data layer for floating-point values.

    Parameters
    ----------
    data : float, optional
        The value.
    name : str, default="Float"
        Name of the layer.
    min : float, optional
        Minimum accepted value.
    max : float, optional
        Maximum accepted value.
    step : float, default=0.1
        Step size of the spin box in user interfaces.
    default : float, default=0.0
        Default value, used when `data` is not provided.
    auto_call : bool, default=False
        Re-run the algorithm when the value changes in user interfaces.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).

    Examples
    --------
    >>> sigma = sk.Float(name="Sigma", min=0, default=1.0, step=0.5)
    """

    kind = "float"
    type = Optional[float]

    def __init__(
        self,
        data: Optional[float] = None,
        name="Float",
        min: float = float(np.finfo(np.float32).min),
        max: float = float(np.finfo(np.float32).max),
        step: float = 0.1,
        default: float = 0.0,
        required: bool = True,
        auto_call: bool = False,
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
        if value_range != (float(np.finfo(np.float32).min), float(np.finfo(np.float32).max)):
            summary += f" in [{fmt_num(value_range[0])}, {fmt_num(value_range[1])}]"
        return summary
