"""
Documentation: https://imaging-server-kit.github.io/imaging-server-kit/
"""

from pathlib import Path

import imaging_server_kit as sk

# Import your package if needed (also add it to requirements.txt)
# (...)

@sk.algorithm(
    name="{{ cookiecutter.name }}",
    description="",
    project_url="{{ cookiecutter.project_url }}",
    tags=["Segmentation"],
    parameters={
        "image": sk.Image(),
        "threshold": sk.Float(
            name="Threshold",
            default=128,
            min=0,
            max=255,
            step=1,
            auto_call=False,
        ),
    },
    samples=[
        {
            "image": Path(__file__).parent / "samples" / "blobs.tif",
            "threshold": 140,
        }
    ],
)
def threshold_algo(image, threshold):
    segmentation = image > threshold  # Replace this with your code
    return sk.Mask(segmentation, name="Binarized image")


if __name__ == "__main__":
    sk.serve(threshold_algo)  # Serve on http://localhost:8000
