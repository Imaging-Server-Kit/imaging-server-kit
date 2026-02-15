from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer


class Notification(DataLayer):
    """Data layer used to represent a text notification.

    Use the `level` parameter to define the notification level (`info`, `warning`, or `error`).

    Example:
        notif = sk.Notification("Warning!", level="warning")
    """

    kind = "notification"
    type = str

    def __init__(
        self,
        data: Optional[str] = None,
        level: Optional[str] = "info",
        name="Notification",
        description="Text notification",
        default: Optional[str] = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
            default=default,
            level=level,
            **kwargs,
        )

    def __str__(self) -> str:
        level = self.meta.get("level", "info")
        return f"{self.name} ({level}): {self.data}"

    def refresh(self):
        print(self)
