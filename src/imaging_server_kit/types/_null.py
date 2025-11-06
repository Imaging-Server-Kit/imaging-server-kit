from typing import Any, Dict, Optional
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
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.default = default
        
        # Schema contributions
        main = {"default": self.default}
        extra = {}
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    @classmethod
    def serialize(cls, data, client_origin: str):
        if data is not None:
            raise ValueError(f"Cannot serialize this object: {data}")
        return None

    @classmethod
    def deserialize(cls, serialized_data, client_origin: str):
        return None
