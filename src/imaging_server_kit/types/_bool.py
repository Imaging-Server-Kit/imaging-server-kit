from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class Bool(DataLayer):
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
        self.kind = "bool"
        self.type = bool
        self.default = default
        self.auto_call = auto_call

    @classmethod
    def to_features(cls, data):
        return bool(data)

    @classmethod
    def to_data(cls, features):
        return bool(features)
