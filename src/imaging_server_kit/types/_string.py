from typing import Optional

from imaging_server_kit.types.layer import Layer


class String(Layer):
    """Data layer for strings of text.

    Parameters
    ----------
    data : str, optional
        The text.
    name : str, default="String"
        Name of the layer.
    default : str, default=""
        Default value, used when `data` is not provided.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer], e.g. `position`, `meta`, or
        extra metadata such as display properties (`colormap="viridis"`).
    """

    kind = "str"
    type = Optional[str]

    def __init__(
        self,
        data: Optional[str] = None,
        name="String",
        default: str = "",
        required: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            default=default,
            required=required,
            **kwargs,
        )
