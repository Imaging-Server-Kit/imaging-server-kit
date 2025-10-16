"""
Algorithm server definition.
Documentation: https://imaging-server-kit.github.io/imaging-server-kit/
"""

from pathlib import Path

import imaging_server_kit as sk

# Import your package if needed (also add it to requirements.txt)
# import [...]


@sk.algorithm(
    name="{{ cookiecutter.name }}",
    title="{{ cookiecutter.title }}",
    description="",
    project_url="{{ cookiecutter.project_url }}",
    tags=["Segmentation"],
    parameters={
        "image": sk.Image(dimensionality=[2, 3]),
        "threshold": sk.Float(
            name="Threshold",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            auto_call=True,
        ),
    },
    samples=[
        {
            "image": str(Path(__file__).parent / "sample_images" / "blobs.tif"),
            "threshold": 0.7,
        },
        {
            "image": "https://github.com/Imaging-Server-Kit/imaging-server-kit/blob/main/src/imaging_server_kit/demo/sample_images/blobs.tif"
        }
    ],
)
def threshold_sk(image, threshold):
    segmentation = image > threshold  # Replace this with your code
    return sk.Mask(data=segmentation, name="Binarized image")


if __name__ == "__main__":
    sk.serve(threshold_sk)  # Serve on http://localhost:8000
