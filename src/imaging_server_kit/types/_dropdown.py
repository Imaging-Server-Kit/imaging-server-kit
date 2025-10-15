from typing import Dict, List, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from imaging_server_kit.types.data_layer import DataLayer


class DropDown(DataLayer):
    def __init__(
        self,
        data: Optional[List] = None,
        name="Choice",
        description="Dropdown selection",
        items: Optional[List] = None,
        default: str = None,
        auto_call: bool = False,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        if items is None:
            items = []
        self.kind = "dropdown"
        self.type = Literal.__getitem__(tuple(items))
        self.default = default
        self.auto_call = auto_call

    @classmethod
    def to_features(cls, data):
        return str(data)

    @classmethod
    def to_data(cls, features):
        return str(features)
