from typing import Dict, Optional

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

        # TODO: implement string-specific properties, for example: max_length (could be validated).

    @classmethod
    def serialize(cls, data: Optional[str], client_origin: str) -> Optional[str]:
        if data is not None:
            return str(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[str], client_origin: str) -> Optional[str]:
        if serialized_data is not None:
            return str(serialized_data)
