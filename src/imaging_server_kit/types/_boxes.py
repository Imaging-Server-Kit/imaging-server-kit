from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np
from geojson import Feature, Polygon

from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.types.common import (
    extract_meta_tile,
    ObjectMerger,
    ObjectTileMerger,
)
from imaging_server_kit.types.data_serializer import DataSerializer
from imaging_server_kit.types.data_layer import DataLayer, Merger


def _get_tile(boxes: Boxes, tile_meta: TileMeta):
    # Mask of box coordinates in the tile
    boxes_coords_in_tile = (boxes.data >= tile_meta.coords_min) & (
        boxes.data < tile_meta.coords_max
    )

    # All coordinates must be in the tile bounds
    tile_filter = boxes_coords_in_tile.reshape((len(boxes_coords_in_tile), -1)).all(
        axis=1
    )

    # Select boxes in the tile
    boxes_data_tile = boxes.data[tile_filter]

    # Select meta of boxes in the tile
    boxes_meta_tile = extract_meta_tile(boxes.meta, boxes.n_objects, tile_filter)

    return boxes_data_tile, boxes_meta_tile, tile_filter


class BoxesDataSerializer(DataSerializer):
    def serialize(
        self, boxes: Optional[np.ndarray], client_origin: str
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

    def deserialize(
        self, serialized_boxes: Optional[List[Feature]], client_origin: str
    ) -> Optional[np.ndarray]:
        if serialized_boxes is None:
            return None
        boxes = np.array(
            [feature["geometry"]["coordinates"] for feature in serialized_boxes]
        )
        boxes = np.array(
            [box[0] for box in boxes]
        )  # We get back a shape of (N, 1, 5, 2) - so we remove dim. 1
        if len(boxes) > 0:
            boxes = boxes[:, :-1]  # Remove the last element
            boxes = boxes[:, :, ::-1]  # Invert XY
        return boxes


class Boxes(DataLayer):
    """Data layer used to represent boxes (rectangular bounding boxes).

    Parameters
    ----------
    data: A Numpy array of shape (N, 4, D) containing the coordinates of the four corners of the box.
    """

    kind = "boxes"
    mergers: Dict[str, Type[Merger]] = {
        "default": ObjectTileMerger,
        "override": ObjectMerger,
    }
    data_serializers: Dict[str, Type[DataSerializer]] = {"default": BoxesDataSerializer}

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        name="Boxes",
        description="Bounding boxes.",
        dimensionality: Optional[List[int]] = None,
        required: bool = True,
        merger: str = "default",
        data_serializer: str = "default",
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
            dimensionality=dimensionality,
            required=required,
            merger=merger,
            data_serializer=data_serializer,
            **kwargs,
        )

    @property
    def data_global_coords(self) -> Optional[np.ndarray]:
        """Data in global coordinate reference instead of local to the tile."""
        if (self.data is not None) and (self.tile_meta is not None):
            if self.tile_meta.coords_min is not None:
                return self.data + self.tile_meta.coords_min

    @property
    def n_objects(self) -> int:
        if self.data is None:
            return 0
        else:
            return len(self.data)

    @property
    def data_pixel_domain(self) -> Optional[Tuple]:
        if self.data is not None:
            if self.n_objects > 0:
                return tuple(np.max(np.asarray(self.data), axis=(0, 1)))

    def get_tile(self, tile_meta: TileMeta) -> Boxes:
        if self.data is None:
            _data = self.data
            _meta = self.meta
        if self.n_objects == 0:
            _data = self._get_initial_data(self.pixel_domain)
            _meta = self.meta
        else:
            boxes_tile_data, boxes_tile_meta, _ = _get_tile(self, tile_meta)
            if boxes_tile_data is not None:
                boxes_tile_data = boxes_tile_data - tile_meta.coords_min
            _data = boxes_tile_data
            _meta = boxes_tile_meta
        return Boxes(
            data=_data,
            name=self.name,
            meta=_meta,
            tile_meta=tile_meta,
        )

    @staticmethod
    def _get_initial_data(
        pixel_domain: Optional[Union[Tuple, List]],
    ) -> Optional[np.ndarray]:
        if pixel_domain is None:
            return
        return np.zeros((0, 4, len(pixel_domain)), dtype=np.float32)

    @staticmethod
    def validate_data(data, meta):
        if meta["required"] is False:
            return

        assert isinstance(
            data, np.ndarray
        ), f"Boxes data ({type(data)}) is not a Numpy array"

        assert len(data.shape) == 3, "Boxes data should have shape (N, 4, D)"

        allowed_dims = meta["dimensionality"]
        assert (
            data.shape[2] in allowed_dims
        ), f"Boxes have an unsupported dimensionality: {data.shape[2]} (accepted: {allowed_dims})"
