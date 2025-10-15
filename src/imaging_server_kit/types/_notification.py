from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class Notification(DataLayer):
    def __init__(
        self,
        data: Optional[str] = None,
        name="Notification",
        description="Text notification",
        default: str = None,
        meta: Optional[Dict] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
        )
        self.kind = "notification"
        self.type = str
        self.default = default

    @classmethod
    def to_features(cls, data):
        return str(data)

    @classmethod
    def to_data(cls, features):
        return str(features)

    def __str__(self) -> str:
        level = self.meta.get("level", "info")
        return f"Notification ({level}): {self.data}"

    def refresh(self):
        print(self)
