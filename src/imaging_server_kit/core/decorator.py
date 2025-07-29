from functools import partial
from inspect import isgeneratorfunction
from typing import Callable, List

import numpy as np
import skimage.io
from pydantic import (BaseModel, ConfigDict, Field, create_model,
                      field_validator)

from .encoding import decode_contents
from .server import AlgorithmServer


class BaseParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

def decode_image_array(cls, v, dimensionality) -> "np.ndarray":
    image_array = decode_contents(v)

    if image_array.ndim not in dimensionality:
        raise ValueError("Array has the wrong dimensionality.")

    return image_array

def decode_points_array(cls, v, dimensionality) -> "np.ndarray":
    points_array = decode_contents(v)

    if points_array.shape[1] not in dimensionality:
        raise ValueError("Array has the wrong dimensionality.")

    return points_array

def decode_generic(cls, v, dimensionality) -> "np.ndarray":
    # TODO: there should also be functions to validate dimensionality for tracks and vectors.
    return decode_contents(v)

def parse_params(parameters: dict) -> BaseModel:
    fields = {}
    validators = {}

    if parameters is not None:
        for param_name, param_details in parameters.items():
            field_constraints = {"json_schema_extra": {}}

            if hasattr(param_details, "min"):
                field_constraints["ge"] = param_details.min
            if hasattr(param_details, "max"):
                field_constraints["le"] = param_details.max
            if hasattr(param_details, "default"):
                field_constraints["default"] = param_details.default
                field_constraints["example"] = param_details.default
            if hasattr(param_details, "title"):
                field_constraints["title"] = param_details.title
            if hasattr(param_details, "description"):
                field_constraints["description"] = param_details.description
            if hasattr(param_details, "step"):
                field_constraints["json_schema_extra"]["step"] = param_details.step
            if hasattr(param_details, "auto_call"):
                field_constraints["json_schema_extra"]["auto_call"] = param_details.auto_call

            field_constraints["json_schema_extra"][
                "widget_type"
            ] = param_details.widget_type

            if param_details.widget_type in ["image", "mask", "mask3d", "instance_mask"]:
                validated_func = partial(
                    decode_image_array, dimensionality=param_details.dimensionality
                )
                validator_name = f"validate_{param_name}_image"
                validators[validator_name] = field_validator(param_name, mode="after")(
                    validated_func
                )
                # TODO: Another validator to check for param_details.rgb if it's an `image`
                # ...
            elif param_details.widget_type == "points":
                validated_func = partial(
                    decode_points_array, dimensionality=param_details.dimensionality
                )
                validator_name = f"validate_{param_name}_image"
                validators[validator_name] = field_validator(param_name, mode="after")(
                    validated_func
                )
            elif param_details.widget_type in ["vectors", "tracks", "shapes"]:
                validated_func = partial(
                    decode_generic, dimensionality=param_details.dimensionality
                )
                validator_name = f"validate_{param_name}_image"
                validators[validator_name] = field_validator(param_name, mode="after")(
                    validated_func
                )
            
            fields[param_name] = (param_details.type, Field(**field_constraints))

    return create_model(
        "Parameters",
        __base__=BaseParamsModel,
        __validators__=validators,
        **fields,
    )


class CustomAlgorithmServer(AlgorithmServer):
    def __init__(
        self,
        parameters_model,
        algorithm_name,
        title,
        description,
        used_for,
        tags,
        project_url,
        serverkit_repo_url,
        metadata_file,
        func: Callable,
        sample_images: List,
    ):
        super().__init__(
            parameters_model=parameters_model,
            algorithm_name=algorithm_name,
            metadata_file=metadata_file,
            title=title,
            description=description,
            tags=tags,
            used_for=used_for,
            project_url=project_url,
            serverkit_repo_url=serverkit_repo_url,
        )
        self.run_algorithm = func  # `Rename` the function
        self.stream = isgeneratorfunction(func)
        self.sample_images = sample_images

    def load_sample_images(self) -> List["np.ndarray"]:
        # Sample images can be image paths, URLs or Numpy arrays
        images = []
        for sample_image_or_path in self.sample_images:
            if isinstance(sample_image_or_path, np.ndarray):
                images.append(sample_image_or_path)
            else:
                images.append(skimage.io.imread(sample_image_or_path))
        return images


def algorithm_server(
    parameters=None,
    algorithm_name="algorithm",
    title="Image Processing Algorithm",
    description="Implementation of an image processing algorithm.",
    used_for=[],
    tags=[],
    project_url="https://github.com/Imaging-Server-Kit/imaging-server-kit",
    sample_images=[],
    metadata_file: str = None,
    serverkit_repo_url="https://github.com/Imaging-Server-Kit/imaging-server-kit",
):
    """
    This decorator wraps a function and converts it to an algorithm server.
    Args:
        parameters: A dictionary defining the parameters for the algorithm.
        algorithm_name: The name of the algorithm, in URL-friendly format (no spaces or special characters).
        title: A title or display name for the algorithm. Used in the /info page.
        description: A description of the algorithm. Used in the /info page.
        used_for: A list of tags from: `Segmentation`, `Registration`, `Filtering`, `Tracking`, `Restoration`, `Points detection`, `Box detection`.
        tags: Tags used to categorize the algorithm server, for example `Deep learning`, `EPFL`, `Cell biology`, `Digital pathology`.
        project_url: A URL to the project homepage.
        sample_images: A list of paths to sample images for the algorithm.
    """
    def wrapper(func: Callable):
        algo_server = CustomAlgorithmServer(
            parameters_model=parse_params(parameters),
            algorithm_name=algorithm_name,
            title=title,
            description=description,
            used_for=used_for,
            tags=tags,
            project_url=project_url,
            serverkit_repo_url=serverkit_repo_url,
            metadata_file=metadata_file,
            func=func,
            sample_images=sample_images,
        )

        return algo_server

    return wrapper
