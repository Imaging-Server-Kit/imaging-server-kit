from pathlib import Path
from typing import List
import os

import numpy as np
import uvicorn
from cellpose import models

from imaging_server_kit import (
    DropDownUI,
    FloatUI,
    ImageUI,
    IntUI,
    algorithm_server,
)

custom_model_path = "/models"  # Use relative path for portability
if os.path.exists(custom_model_path):
    custom_models = [f.name for f in os.scandir(custom_model_path) if f.is_file()]
else:
    custom_models = []
base_models = ["cpsam"]  # Add all built-in models you want to support
models_list = base_models + custom_models
print(f"Available models: {models_list}")

# Global cache variables for performance
last_model = None
cached_model = None
last_custom_path = None
cached_custom_model = None

@algorithm_server(
    algorithm_name="cellpose",
    parameters={
        "image": ImageUI(
            title="Image",
            description="Input image (2D).",
            dimensionality=[2],
        ),
        "model_name": DropDownUI(
            default="cpsam",
            title="Model",
            description="The model used for instance segmentation",
            items=models_list,  # Use dynamic list
        ),
        "flow_threshold": FloatUI(
            default=0.3,
            title="Flow threshold",
            description="The flow threshold",
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        "cellprob_threshold": FloatUI(
            default=0.5,
            title="Probability threshold",
            description="The detection probability threshold",
            min=0.0,
            max=1.0,
            step=0.01,
        ),
    },
    title="CellPose",
    description="A generalist algorithm for cellular segmentation.",
    used_for=["Segmentation"],
    tags=[
        "Deep learning",
        "Fluorescence microscopy",
        "Digital pathology",
        "Cell biology",
    ],
    project_url="https://github.com/MouseLand/cellpose",
    sample_images=[
        str(Path(__file__).parent / "sample_images" / "nuclei_2d.tif"),
    ],
)
def cellpose_server(
    image: np.ndarray,
    model_name: str,
    flow_threshold: float,
    cellprob_threshold: float,
) -> List[tuple]:
    """Runs the algorithm."""
    global last_model, cached_model, last_custom_path, cached_custom_model
    if len(image.shape) != 2:
        if image.shape[0] == 1:
            image = image[0]
        else: 
            raise ValueError("Input image must be 2D")
    if model_name in custom_models:
        model_path = os.path.join(custom_model_path, model_name)
        if model_name != last_custom_path:
            print(f"Loading custom model from {model_path}")
            cached_custom_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
            last_custom_path = model_name
        else:
            print(f"Using cached custom model from {model_path}")
        model = cached_custom_model
    else:
        if model_name != last_model:
            print("Loading built-in model")
            cached_model = models.CellposeModel(gpu=True, pretrained_model=model_name)
            last_model = model_name
        else:
            print("Using cached built-in model")
        model = cached_model

    segmentation, flows, styles = model.eval(
        image,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    segmentation_params = {"name": "Cellpose result"}

    return [
        (segmentation, segmentation_params, "instance_mask"),
    ]


if __name__ == "__main__":
    uvicorn.run(cellpose_server.app, host="0.0.0.0", port=8000)
