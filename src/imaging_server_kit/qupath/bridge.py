"""
Experimental bridge between QuPath and the Imaging Server Kit via QuBaLab.

This method is likely going to replace the qupath-extension-serverkit in the future.

Currently, the bridge is implemented for algorithms returning sk.Mask and sk.Boxes objects.
"""

from typing import List, Optional, Union

import geojson
import numpy as np
import qubalab.qupath as qp
from qubalab.images import QuPathServer
from qubalab.objects import ObjectType
from geojson import Feature, Polygon
from shapely.geometry import shape

import imaging_server_kit as sk
from imaging_server_kit.types._mask import mask2features
from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.algorithm import Algorithm
from imaging_server_kit.core.multialgo import MultiAlgorithm
from imaging_server_kit.remote import Client

from tqdm import tqdm


def _mask2detections(mask: sk.Mask) -> List[Feature]:
    """Convert a Mask object to a list of GeoJson features for QuPath."""
    detections = []

    features = mask2features(mask.data)

    if len(features) == 0:
        return detections

    mask_features = mask.meta.get("features")
    if mask_features:
        feature_classes = mask_features.get("class")
    else:
        feature_classes = None

    for f in features:
        f["object_type"] = "detection"

        if feature_classes:
            f["classification"] = {
                "name": feature_classes,
                "color": (0, 255, 0),
            }

        feature_geom = np.array(f["geometry"]["coordinates"])
        feature_geom = feature_geom[0]

        # Global offset (coords have to be inverted, for some reason)
        feature_geom[:, 0] = feature_geom[:, 0] + mask.position[1]
        feature_geom[:, 1] = feature_geom[:, 1] + mask.position[0]
        f["geometry"]["coordinates"] = feature_geom[None].tolist()

        detections.append(f)

    return detections


def _boxes2detections(boxes: sk.Boxes) -> List[Feature]:
    """Convert a Boxes object to a list of GeoJson features for QuPath."""
    detections = []

    if boxes.n_objects == 0:
        return detections

    mask_features = boxes.meta.get("features")
    if mask_features:
        feature_classes = mask_features.get("class")
    else:
        feature_classes = None

    for box_data in boxes.data_global_coords:
        box_data_closed = np.vstack(
            [box_data, box_data[0]]
        )  # Close the polygon for QuPath
        coords = np.asarray([box_data_closed.tolist()])
        coords = coords[:, :, ::-1]  # Invert X-Y
        try:
            geom = Polygon(coordinates=coords.tolist(), validate=True)
            f = Feature(geometry=geom)
            f["object_type"] = "detection"

            if feature_classes:
                f["classification"] = {
                    "name": feature_classes,
                    "color": (0, 255, 0),
                }

            detections.append(f)
        except ValueError:
            print("Invalid polygon geometry.")

    return detections


def _get_annotations_by_class_name(
    annotations: List[geojson.Feature], cls_name: str
) -> List[geojson.Feature]:
    qp_image_roi_annotations = []

    for ann in annotations:
        if ann.classification is not None:
            if ann.classification.names[0] == cls_name:
                qp_image_roi_annotations.append(ann)

    return qp_image_roi_annotations


def run_in_qupath_annotations(
    runner: Union[Algorithm, MultiAlgorithm, Client, AlgorithmRunner],
    annotation_name: str = "Region",
    token: str = "",
    port: int = 25333,
    tiled=False,
    tile_size=64,
    tile_overlap=0,
    algorithm_name: Optional[str] = None,
    **algo_params,
):
    """
    Experimental bridge between QuPath and the Imaging Server Kit via QuBaLab.
    Access image data inside QuPath annotations with a given name (highest resolution available),
    run an algorithm, and return the results as QuPath detections.
    
    Parameters
    ----------
    runner: A server kit algorithm, multi-algorithm, or client object.
    annotation_name: Name of the QuPath annotations to run the algorithm into.
    token: Token from the Py4J extension.
    port: Port from the Py4J extension.
    tiled: Set to True to enable tiled inference in the annotations.
    tile_size: Tile size in pixels.
    tile_overlap: Relative overlap between tiles.
    algorithm_name: Optional name of the algorithm to run (e.g. when using a sk.Client or MultiAlgorithm).
    **algo_params: Parameters of the algorithm to run.
    """
    gateway = qp.create_gateway(auth_token=token, port=port)
    server = QuPathServer(gateway)

    qp_annotations = qp.get_objects(
        object_type=ObjectType.ANNOTATION, converter="geojson"
    )

    found_annotations = _get_annotations_by_class_name(
        qp_annotations, cls_name=annotation_name
    )

    for annotation in found_annotations:
        bounds = shape(annotation.geometry).bounds

        min_x = int(max(0, bounds[0]))
        min_y = int(max(0, bounds[1]))
        max_x = int(min(server.metadata.width, bounds[2]))
        max_y = int(min(server.metadata.height, bounds[3]))

        # Get a sk.Domain from QuBalab's retreived annotation
        domain = sk.Domain(position=(min_x, min_y), size=(max_x - min_x, max_y - min_y))

        if tiled is False:
            tile_size = domain.size

        # Generate tiles
        result = sk.Stack()

        pbar = tqdm(desc="Processing tiles")
        for tile_meta, tile_domain in sk.generate_tiles(
            domain, tile_size=tile_size, tile_overlap=tile_overlap
        ):
            # Update a progress bar
            pbar.total = tile_meta.n_tiles
            pbar.n = tile_meta.tile_idx + 1
            pbar.refresh()

            # Read the tile region
            image_tile = server.read_region(
                x=tile_domain.coords_min[0],
                y=tile_domain.coords_min[1],
                width=tile_domain.size[0],
                height=tile_domain.size[1],
            )

            # Handle RGB case
            if server.metadata.is_rgb:
                image_tile = np.moveaxis(image_tile, 0, -1)

            # Run the algo (in regular mode)
            tile_result = runner.run(
                image=image_tile, algorithm=algorithm_name, **algo_params
            )

            # Offset stack position (coords have to be inverted, for some reason)
            tile_result.position = (
                tile_domain.coords_min[1],
                tile_domain.coords_min[0],
            )

            # Merge the tile into a stack
            result.merge(tile_result)

        pbar.close()

        # Send back compatible results (Mask or Boxes) to QuPath
        detections = []
        for layer in result:
            if isinstance(layer, sk.Mask):
                mask_detections = _mask2detections(layer)
                detections.extend(mask_detections)
            elif isinstance(layer, sk.Boxes):
                print("We're converting boxes!")
                box_detections = _boxes2detections(layer)
                detections.extend(box_detections)

        if len(detections) > 0:
            qp.add_objects(
                detections,
                gateway=gateway,
                image_data=qp.get_current_image_data(gateway),
            )

            qp.refresh_qupath(gateway=gateway)
