from typing import Optional

from imaging_server_kit.types.data_layer import DataLayer


class String(DataLayer):
    """Data layer used to represent strings of text."""

    kind = "str"
    type = str

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