from __future__ import annotations

from typing import Dict, Optional

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.data_layer import DataLayer

from tqdm import tqdm

class Progress(DataLayer):
    """Data layer used to represent a progress bar.

    Example:
        notif = sk.Progress()
    """

    kind = "progress"
    type = int

    def __init__(
        self,
        data: Optional[int] = None,
        name="Progress",
        description="Progress bar",
        default: Optional[int] = None,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        # Initialize max_val
        if meta is None:
            meta = {"max_val": 1}
        else:
            if "max_val" not in meta:
                meta["max_val"] = 1
        
        if data is None:
            data = 0
        
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
        )
        self.default = default
        
        # Schema contributions
        main = {"default": self.default}
        extra = {}
        self.constraints = [main, extra]
        
        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)
        
        # TQDM progress bar
        self.pbar = tqdm()
        self.refresh()

    @classmethod
    def serialize(cls, data: Optional[int], client_origin: str) -> Optional[int]:
        if data is not None:
            return int(data)

    @classmethod
    def deserialize(cls, serialized_data: Optional[int], client_origin: str) -> Optional[int]:
        if serialized_data is not None:
            return int(serialized_data)

    def __str__(self) -> str:
        max_val = self.meta["max_val"]
        return f"Progress (current: {self.data}/{max_val})"

    def refresh(self):
        if self.data is not None:
            self.pbar.total = self.meta["max_val"]
            self.pbar.n = self.data
            self.pbar.refresh()