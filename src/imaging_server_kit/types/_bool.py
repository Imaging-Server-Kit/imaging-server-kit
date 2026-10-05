from typing import Optional

from imaging_server_kit.types.layer import Layer


class Bool(Layer):
    """Data layer for boolean values, shown as a checkbox in user interfaces.

    Parameters
    ----------
    data : bool, optional
        The value.
    name : str, default="Bool"
        Name of the layer.
    default : bool, default=False
        Default value, used when `data` is not provided.
    auto_call : bool, default=False
        Re-run the algorithm when the value changes in user interfaces.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
    """

    kind = "bool"
    type = Optional[bool]

    def __init__(
        self,
        data: Optional[bool] = None,
        name="Bool",
        default: bool = False,
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
            **kwargs,
        )
