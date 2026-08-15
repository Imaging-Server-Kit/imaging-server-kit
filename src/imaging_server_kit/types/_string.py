from typing import Optional

from imaging_server_kit.types.layer import Layer


class String(Layer):
    """Data layer used to represent strings of text.

    Parameters
    ----------
    data: A string value.
    default: Default value used when `data` is not provided.
    """

    kind = "str"
    type = Optional[str]

    def __init__(
        self,
        data: Optional[str] = None,
        name="String",
        description="String parameter",
        default: str = "",
        required: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            description=description,
            default=default,
            required=required,
            **kwargs,
        )
