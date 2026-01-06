from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from geojson import Feature, Polygon

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.common import extract_meta_tile, merge_meta_tile
from imaging_server_kit.types.data_layer import DataLayer


def _get_tile(boxes: Boxes, tile_meta: TileMeta):
    # Mask of box coordinates in the tile
    boxes_coords_in_tile = (boxes.data >= tile_meta.coords_min) & (
        boxes.data < tile_meta.coords_max
    )

    # All coordinates must be in the tile bounds
    tile_filter = boxes_coords_in_tile.reshape((len(boxes_coords_in_tile), -1)).all(axis=1)

    # Select boxes in the tile
    boxes_data_tile = boxes.data[tile_filter]

    # Select meta of boxes in the tile
    boxes_meta_tile = extract_meta_tile(boxes.meta, boxes.n_objects, tile_filter)

    return boxes_data_tile, boxes_meta_tile, tile_filter


class Boxes(DataLayer):
    """Data layer used to represent boxes (rectangular bounding boxes).

    Parameters
    ----------
    data: A Numpy array of shape (N, 4, D) containing the coordinates of the four corners of the box.
    """

    kind = "boxes"

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Boxes",
        description="Bounding boxes.",
        dimensionality: Optional[List[int]] = None,
        required: bool = True,
        meta: Optional[Dict] = None,
        tile_meta: Optional[TileMeta] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            meta=meta,
            data=data,
            tile_meta=tile_meta,
        )
        self.dimensionality = (
            dimensionality if dimensionality is not None else np.arange(6).tolist()
        )
        self.required = required

        # Schema contributions
        main = {}
        if not self.required:
            self.default = None
            main["default"] = self.default
        extra = {
            "dimensionality": self.dimensionality,
            "required": self.required,
        }
        self.constraints = [main, extra]

        if self.data is not None:
            self.validate_data(data, self.meta, self.constraints)

        # TODO: Implement object-specific properties, like max_objects or min_box_area (could be validated).

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def _pixel_domain(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(np.asarray(self.data), axis=(0, 1)))

    def get_tile(self, tile_meta: TileMeta) -> Boxes:
        if self.data is None:
            return Boxes(
                data=self.data,
                name=self.name,
                meta=self.meta,
                tile_meta=tile_meta,
            )
        if self.n_objects == 0:
            return Boxes(
                data=self._get_initial_data(self.pixel_domain), # type: ignore
                name=self.name,
                meta=self.meta,
                tile_meta=tile_meta,
            )
        else:
            boxes_tile_data, boxes_tile_meta, _ = _get_tile(self, tile_meta)
            if boxes_tile_data is not None:
                boxes_tile_data = boxes_tile_data - tile_meta.coords_min
            return Boxes(
                data=boxes_tile_data,
                name=self.name,
                meta=boxes_tile_meta,
                tile_meta=tile_meta,
            )

    def merge_tile(self, boxes_tile: Boxes) -> None:
        if (boxes_tile.data is None) or (boxes_tile.tile_meta is None):
            raise RuntimeError("Invalid attempt to merge a box tile.")

        if self.n_objects > 0:
            # Offset the tile data by the tile positions
            boxes_tile.data = boxes_tile.data + boxes_tile.tile_meta.coords_min

            # Remove the previous tile boxes from the data boxes
            *_, tile_filter = _get_tile(self, boxes_tile.tile_meta)
            boxes_clean = self.data[~tile_filter]

            # Merge the new tile boxes with the data boxes
            merged_boxes_data = np.vstack((boxes_clean, boxes_tile.data)) # type: ignore

            # Do the same for boxes metadata
            merged_boxes_meta = merge_meta_tile(
                self.meta, boxes_tile.meta, self.n_objects, tile_filter
            )
        else:
            merged_boxes_data = boxes_tile.data
            merged_boxes_meta = boxes_tile.meta

        self.data = merged_boxes_data
        self.meta = merged_boxes_meta

    @classmethod
    def serialize(
        cls, boxes: Optional[np.ndarray], client_origin: str
    ) -> Optional[List[Feature]]:
        if boxes is None:
            return None

        features = []
        for i, box in enumerate(boxes):
            coords = np.array(box)[:, ::-1]  # Invert XY
            coords = coords.tolist()
            coords.append(coords[0])  # Close the Polygon
            try:
                geom = Polygon(coordinates=[coords], validate=True)
                features.append(Feature(geometry=geom, properties={"Detection ID": i}))
            except ValueError:
                print(
                    "Invalid box polygon geometry. Expected an array of shape (N, 4, D) representing the corners of the box."
                )
        return features

    @classmethod
    def deserialize(
        cls, serialized_data: Optional[List[Dict[str, Any]]], client_origin: str
    ) -> Optional[np.ndarray]:
        if serialized_data is None:
            return None
        boxes = np.array(
            [feature["geometry"]["coordinates"] for feature in serialized_data]
        )
        boxes = np.array(
            [box[0] for box in boxes]
        )  # We get back a shape of (N, 1, 5, 2) - so we remove dim. 1
        if len(boxes) > 0:
            boxes = boxes[:, :-1]  # Remove the last element
            boxes = boxes[:, :, ::-1]  # Invert XY
        return boxes

    @classmethod
    def _get_initial_data(
        cls, pixel_domain: Optional[Union[Tuple, List]]
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((0, 4, len(pixel_domain)), dtype=np.float32)

    @classmethod
    def validate_data(cls, data, meta, constraints):
        main, extra = constraints
        if extra["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Boxes data ({type(data)}) is not a Numpy array"

        assert len(data.shape) == 3, "Boxes data should have shape (N, 4, D)"

        allowed_dims = extra["dimensionality"]
        assert (
            data.shape[2] in allowed_dims
        ), f"Boxes have an unsupported dimensionality: {data.shape[2]} (accepted: {allowed_dims})"
