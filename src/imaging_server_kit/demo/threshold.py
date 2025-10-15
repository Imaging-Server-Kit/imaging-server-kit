from pathlib import Path
import numpy as np
from skimage.exposure import rescale_intensity

import imaging_server_kit as sk


@sk.algorithm(
    name="threshold",
    title="Binary Threshold",
    description="Implementation of a binary threshold algorithm.",
    tags=["Demo", "Segmentation"],
    parameters={
        "image": sk.Image(),
        "threshold": sk.Float(
            default=0.5,
            name="Threshold",
            description="Intensity threshold.",
            min=0.0,
            max=1.0,
            step=0.1,
            auto_call=True,
        ),
    },
    sample_images=[str(Path(__file__).parent / "sample_images" / "blobs.tif")],
)
def threshold(image: np.ndarray, threshold: float):
    """Implements a simple intensity threshold algorithm."""
    mask = rescale_intensity(image, out_range=(0, 1)) > threshold
    return sk.Mask(mask, name="Threshold result")


if __name__ == "__main__":
    sk.serve(threshold)