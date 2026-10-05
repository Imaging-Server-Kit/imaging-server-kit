from typing import Optional

from imaging_server_kit.types.layer import Layer


class Notification(Layer):
    """Data layer for text notifications.

    Notifications are printed to the terminal, or shown in user interfaces (as Napari
    notifications, for example).

    Parameters
    ----------
    data : str, optional
        The notification text.
    level : str, default="info"
        Notification level: `"info"`, `"warning"`, or `"error"`.
    name : str, default="Notification"
        Name of the layer.
    **kwargs
        Passed to [`Layer`][imaging_server_kit.Layer].

    Examples
    --------
    >>> notif = sk.Notification("Warning!", level="warning")
    """

    kind = "notification"
    type = Optional[str]

    def __init__(
        self,
        data: Optional[str] = None,
        level: Optional[str] = "info",
        name="Notification",
        required: bool = True,
        default: str = "",
        **kwargs,
    ):
        super().__init__(
            name=name,
            data=data,
            level=level,
            required=required,
            default=default,
            **kwargs,
        )

    def __str__(self) -> str:
        level = self.meta.get("level", "info")
        return f"{self.name} ({level}): {self.data}"

    def _display(self) -> None:
        if self.data is not None:
            print(self)
