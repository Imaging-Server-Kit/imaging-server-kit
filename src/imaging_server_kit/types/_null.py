from typing import Any, Optional
from imaging_server_kit.types.data_layer import DataLayer


class Null(DataLayer):
    """
    Data layer used to represent None or the absence of data.
    """

    kind = "null"
    type = type(None)
    
    def __init__(
        self,
        data: Optional[Any] = None,
        name="None",
        description="Null (None) type",
        default=None,
        serializer: str = "default",
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            default=default,
            serializer=serializer,
            **kwargs,
        )