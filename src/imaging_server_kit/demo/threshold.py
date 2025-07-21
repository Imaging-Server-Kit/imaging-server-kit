from pathlib import Path
import numpy as np
import uvicorn
from imaging_server_kit import algorithm_server, ImageUI, FloatUI
from skimage.exposure import rescale_intensity


@algorithm_server(
    algorithm_name="intensity-threshold",
    title="Binary Threshold",
    description="Implementation of a binary threshold algorithm.",
    used_for=["Segmentation"],
    tags=["Demo"],
    parameters={
        "image": ImageUI(dimensionality=[2, 3]),
        "threshold": FloatUI(
            default=0.5,
            title="Threshold",
            description="Intensity threshold.",
            min=0.0,
            max=1.0,
            step=0.1,
        ),
    },
    sample_images=[str(Path(__file__).parent / "sample_images" / "blobs.tif")],
)
def threshold_algo_server(
    image: np.ndarray,
    threshold: float,
):
    """Implements a simple intensity threshold algorithm."""
    mask = rescale_intensity(image, out_range=(0, 1)) > threshold

    if image.ndim == 2:
        data_type = "mask"
    elif image.ndim == 3:
        data_type = "mask3d"

    return [(mask, {"name": "Threshold result"}, data_type)]


if __name__ == "__main__":
    uvicorn.run(threshold_algo_server.app, host="0.0.0.0", port=8000)
