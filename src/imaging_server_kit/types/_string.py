from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class String(DataLayer):
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
        self.kind = "str"
        self.type = str
        self.default = default

    @classmethod
    def to_features(cls, data):
        return str(data)

    @classmethod
    def to_data(cls, features):
        return str(features)
