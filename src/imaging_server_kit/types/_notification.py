from typing import Optional

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
        required: bool = True,
        default: str = "",
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            data=data,
            level=level,
            required=required,
            default=default,
            **kwargs,
        )

    def __str__(self) -> str:
        level = self.meta.get("level", "info")
        return f"{self.name} ({level}): {self.data}"

    def refresh(self):
        print(self)
