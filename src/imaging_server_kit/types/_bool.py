from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class Bool(DataLayer):
    """Data layer used to represent boolean values."""

    kind = "bool"
    type = bool

    def __init__(
        self,
        data: Optional[bool] = None,
        name="Bool",
        description="Boolean parameter",
        default: bool = False,
        auto_call: bool = False,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.default = default
        self.auto_call = auto_call

        # Schema contributions
        main = {"default": self.default}
        extra = {"auto_call": self.auto_call}
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

    @classmethod
    def serialize(cls, data: Optional[bool], client_origin: str):
        if data is not None:
            return bool(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[str], client_origin: str):
        if serialized_data is not None:
            return bool(serialized_data)
