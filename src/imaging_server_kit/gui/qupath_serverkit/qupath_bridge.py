"""
Experimental bridge between QuPath and the Imaging Server Kit via QuBaLab.

Currently, the bridge is implemented for algorithms returning sk.Mask and sk.Boxes objects, but this could be extended in the future.
"""

from typing import Dict, List, Optional, Tuple

import geojson
import numpy as np
import qubalab.qupath as qp
from qubalab.images import QuPathServer
from qubalab.objects import ObjectType
from geojson import Feature, Polygon
from shapely.geometry import shape

import imaging_server_kit as sk
from imaging_server_kit.types._mask import mask2features, instance_mask2features
from imaging_server_kit.core.runner import AlgorithmRunner
from imaging_server_kit.core.tiling import TilingSpecs


MAX_IMAGE_PIXELS = 1024 * 1024 * 256  # Set arbitrarily...


def _mask2detections(mask: sk.Mask) -> List[Feature]:
    """Convert a Mask object to a list of GeoJson features for QuPath."""
    if mask.meta is None:
        return []
    
    if mask.position is None:
        return []
    
    # TODO: We have an issue with handling instance segmentation algorithms in tiled mode, here.
    # With instance masks the detection ID (label of the mask) changes at the end of the process, on_last_tile. 
    # So, if we stream them into QuPath immediately on tile collected, the object IDs will be wrong, and it's unclear 
    # what to do on_last_tile (we cannot currently erase objects from QuPath).
    # On the other hand, with mask2features on a semantic mask we would get `detection ID` = 1, so that doesn't work either.
    # A possible solution would be to only send detections to QuPath on_last_tile after the correct IDs are resolved,
    # but this is a bit disappointing from the user's perspective (no partial results in QuPath...)
    
    # Distinguish between semantic/instance masks
    if mask.meta["merger"] == "default":  # semantic mask
        features = mask2features(mask.data)
    elif mask.meta["merger"] == "instances":
        features = instance_mask2features(mask.data)

    detections = []
    
    if len(features) == 0:
        return detections

    feature_classes = None
    if isinstance(mask.meta, dict):
        mask_features = mask.meta.get("features")
        if isinstance(mask_features, dict):
            feature_classes = mask_features.get("class")
            feature_keys = list(mask_features.keys())

            for k in feature_keys:
                if len(mask_features[k]) != len(features):
                    print(f"Properties array for key {k} does not match features.")
                    # TODO: Do something, like ignore `k` or pop it from the dict?
                    # ...

    for idx, f in enumerate(features):
        f["object_type"] = "detection"
        
        detection_id = f["properties"]["Detection ID"]

        if feature_classes:
            f["classification"] = {
                "name": feature_classes,
                "color": (0, 255, 0),
            }

        if isinstance(mask_features, dict):
            f["properties"]["measurements"] = [
                {"name": key, "value": mask_features[key][detection_id]}
                for key in feature_keys
            ]
        
        # f["properties"]["name"] = f"Mask - {idx}"

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


def if_compatible_get_qupath_schema(
    runner: AlgorithmRunner, algorithm: str
) -> Tuple[Dict, str]:
    """Check if an algorithm is compatible with a QuPath run (= single image as input + no masks, points, etc.)."""
    schema = runner.get_parameters(algorithm)
    params = schema["properties"]

    image_params = []
    for param_name, param in params.items():
        # Parameters inputs that QuPath won't support (if they are required):
        if (param["required"] is True) and (
            param["param_type"]
            in [
                "mask",
                "paths",
                "boxes",
                "points",
                "vectors",
                "tracks",
            ]
        ):
            param_type = param["param_type"]
            print(
                f"Algorithm `{algorithm}` is incompatible with QuPath (requires a parameter of type `{param_type}` as input)."
            )
            return ({}, "")

        if param["param_type"] == "image":
            image_params.append(param_name)

    if len(image_params) == 1:
        qp_image_param = image_params[0]
        params.pop(qp_image_param)
        schema["properties"] = params

        return (schema, qp_image_param)
    else:
        print(
            f"Algorithm `{algorithm}` is incompatible with QuPath (requires `{len(image_params)}` images as input)."
        )
        return ({}, "")


class QuPathBridge:
    """Experimental bridge between QuPath and the Imaging Server Kit via QuBaLab."""

    def __init__(self) -> None:
        self.gateway = None
        self.server = None

    def connect(self, port: int = 25333, token: str = ""):
        """Connect to QuPath via a Py4J gateway (QuBaLab)."""
        self.gateway = qp.create_gateway(auth_token=token, port=port)
        self.server = QuPathServer(self.gateway)

    def get_annotations(self) -> List[geojson.Feature]:
        return qp.get_objects(object_type=ObjectType.ANNOTATION, converter="geojson")  # type: ignore

    def get_annotation_names(self, annotations: List[geojson.Feature]) -> List[str]:
        cls_names = []
        for ann in annotations:
            if ann.classification is not None:
                ann_cls_names = ann.classification.names
                for ann_cls_name in ann_cls_names:
                    if ann_cls_name not in cls_names:
                        cls_names.append(ann_cls_name)

        if len(cls_names) == 0:
            cls_names.append("")

        return cls_names

    def get_annotations_by_class_name(
        self, annotations: List[geojson.Feature], cls_name: str
    ):
        qp_image_roi_annotations = []
        for ann in annotations:
            if ann.classification is not None:
                if ann.classification.names[0] == cls_name:
                    qp_image_roi_annotations.append(ann)

        return qp_image_roi_annotations

    def get_single_annotation(self, cls_name: str) -> geojson.Feature:
        annotations = self.get_annotations()

        found_annotations = self.get_annotations_by_class_name(
            annotations=annotations, cls_name=cls_name
        )

        if len(found_annotations) == 0:
            raise RuntimeError(
                f"⚠️ Could not find the annotation named `{cls_name}` in QuPath."
            )
        elif len(found_annotations) > 1:
            raise RuntimeError(
                f"⚠️ Multiple annotations named `{cls_name}` found in QuPath (found {len(found_annotations)})."
            )
        else:
            return found_annotations[0]

    def run_in_annotation(
        self,
        runner: AlgorithmRunner,
        annotation: Optional[geojson.Feature] = None,
        algorithm_name: Optional[str] = None,
        tiling_ctx: Optional[TilingSpecs] = None,
        **algo_params,
    ):
        """
        Access image data inside a QuPath annotation and run an algorithm on it.

        Parameters
        ----------
        annotation: Optional QuPath annotation to run the algorithm into. If None are provided, the algorithm is run on the entire image.
        runner: A server kit algorithm, multi-algorithm, or client object.

        token: Token from the Py4J extension.
        port: Port from the Py4J extension.

        algorithm_name: Optional name of the algorithm to run (e.g. when using a sk.Client or MultiAlgorithm).
        **algo_params: Parameters of the algorithm to run.
        """
        if self.server is None:
            return

        algorithm = algorithm_name if algorithm_name is not None else runner.name
        schema, qp_image_param = if_compatible_get_qupath_schema(runner, algorithm)
        if not schema:
            return
        
        if isinstance(annotation, geojson.Feature):
            bounds = shape(annotation.geometry).bounds

            min_x = int(max(0, bounds[0]))
            min_y = int(max(0, bounds[1]))
            max_x = int(min(self.server.metadata.width, bounds[2]))
            max_y = int(min(self.server.metadata.height, bounds[3]))
        
        else:
            # Run on the entire image
            min_x = 0
            min_y = 0
            max_x = self.server.metadata.width
            max_y = self.server.metadata.height

        # Get a sk.Domain from QuBalab's retreived annotation
        domain = sk.Domain(position=(min_y, min_x), size=(max_y - min_y, max_x - min_x))
        if domain.size is None:
            return

        n_z = self.server.metadata.n_z_slices
        n_t = self.server.metadata.n_timepoints
        n_c = self.server.metadata.n_channels
        rgb = self.server.metadata.is_rgb

        # Estimated tile size
        max_tile_size = n_z * n_t * n_c * domain.size[0] * domain.size[1]

        if max_tile_size <= MAX_IMAGE_PIXELS:
            tile_size = domain.size
        else:
            # Reduce the XY tile size so that the whole tile does not exceed the limit (use a square tile)
            tile_size = int(np.floor(np.sqrt(MAX_IMAGE_PIXELS / (n_z * n_t * n_c))))
            print(f"Using tile size: {tile_size}")

        if tiling_ctx is None:
            tiling_ctx = TilingSpecs(tile_size=tile_size)

        # Generate tiles
        for tile_meta, tile_domain in sk.generate_tiles(
            domain,
            tile_size=tiling_ctx.tile_size,
            tile_overlap=tiling_ctx.tile_overlap,
            tile_delay=tiling_ctx.tile_delay,
            tile_randomize=tiling_ctx.tile_randomize,
        ):
            if (tile_domain.size is None) or (tile_domain.coords_min is None):
                continue

            n_x = int(tile_domain.size[1])
            n_y = int(tile_domain.size[0])

            # Read a CTZYX tile using `read_region()`
            image_tile = None
            for t in range(n_t):
                for z in range(n_z):
                    image_tile_tz = self.server.read_region(
                        x=tile_domain.coords_min[1],
                        y=tile_domain.coords_min[0],
                        width=n_x,
                        height=n_y,
                        z=z,
                        t=t,
                    )
                    if image_tile is None:
                        # We discover the array type here
                        image_tile = np.empty(
                            (n_c, n_t, n_z, n_y, n_x), dtype=image_tile_tz.dtype
                        )
                    image_tile[:, t, z] = np.asarray(image_tile_tz)

            # Handle RGB case (put channel axis at the end)
            if rgb:
                image_tile = np.moveaxis(image_tile, 0, -1)

            # Squeeze singleton dimensions
            image_tile = np.squeeze(image_tile)

            # Set the image parameter with the QuPath image
            algo_params[qp_image_param] = image_tile

            # Run the algo
            result_tile = runner.run(algorithm=algorithm_name, **algo_params)

            # Offset stack position
            result_tile.position = tile_domain.coords_min
            result_tile.tile_meta = tile_meta

            yield result_tile, domain

    def merge_with_qupath(self, stack: sk.Stack) -> None:
        """Send back results (Mask or Boxes) to QuPath (TODO: quite unreliable)"""
        detections = []
        for layer in stack:
            if isinstance(layer, sk.Mask):
                mask_detections = _mask2detections(layer)
                detections.extend(mask_detections)
            elif isinstance(layer, sk.Boxes):
                box_detections = _boxes2detections(layer)
                detections.extend(box_detections)

        if len(detections) > 0:
            qp.add_objects(
                detections,
                gateway=self.gateway,
                image_data=qp.get_current_image_data(self.gateway),
            )

            qp.refresh_qupath(gateway=self.gateway)
