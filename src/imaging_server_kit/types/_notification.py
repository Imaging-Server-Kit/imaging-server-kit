from typing import Dict, Optional

from imaging_server_kit.types.data_layer import DataLayer


class Notification(DataLayer):
    """Data layer used to represent a text notification.

    Use the `level` meta field to define the notification level (`info`, `warning`, or `error`).

    Example:
        notif = sk.Notification("Warning!", meta={"level": "warning"})
    """

    kind = "notification"
    type = str

    def __init__(
        self,
        data: Optional[str] = None,
        name="Notification",
        description="Text notification",
        default: Optional[str] = None,
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
    def serialize(cls, data: Optional[str], client_origin: str) -> Optional[str]:
        if data is not None:
            return str(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[str], client_origin: str) -> Optional[str]:
        if serialized_data is not None:
            return str(serialized_data)

    def __str__(self) -> str:
        level = self.meta.get("level", "info")
        return f"Notification ({level}): {self.data}"

    def refresh(self):
        print(self)
