from functools import partial, update_wrapper
from inspect import _empty, isgeneratorfunction, signature
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple,
                    Union, get_args, get_origin)

import numpy as np
import skimage.io
from pydantic import (BaseModel, ConfigDict, Field, ValidationError,
                      create_model, field_validator)

import imaging_server_kit.core._etc as etc
import imaging_server_kit.types as skt
from imaging_server_kit.core.results import Results
from imaging_server_kit.core.runner import (AlgorithmRunner, AlgoStream,
                                            algo_stream_gen,
                                            validate_algorithm)
from imaging_server_kit.types import DATA_TYPES, DataLayer

TYPE_MAPPINGS = {
    int: skt.Integer,
    float: skt.Float,
    bool: skt.Bool,
    str: skt.String,
    np.ndarray: skt.Image,
    # To the exception of `DropDown`, sk types can also be used as hints
    skt.Image: skt.Image,
    skt.Mask: skt.Mask,
    skt.Points: skt.Points,
    skt.Vectors: skt.Vectors,
    skt.Boxes: skt.Boxes,
    skt.Paths: skt.Paths,
    skt.Tracks: skt.Tracks,
    skt.Float: skt.Float,
    skt.Integer: skt.Integer,
    skt.Bool: skt.Bool,
    skt.String: skt.String,
    # skt.DropDown: skt.DropDown,  # Won't work
    skt.Notification: skt.Notification,
}

RECOGNIZED_TYPES = tuple(TYPE_MAPPINGS.keys())


class Parameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _parse_run_func_signature(
    func: Callable, parameters: Dict[str, DataLayer]
) -> Dict[str, DataLayer]:
    """Resolve parameters into {param_name : DataLayer} based on the annotations from the decorator and the algo run function signature."""
    def get_data_layer_type(hinted_type, default, param_name) -> DataLayer:
        layer_class = TYPE_MAPPINGS[hinted_type]
        if default is _empty:
            layer = layer_class(name=param_name)
        else:
            layer = layer_class(default=default, name=param_name)
        return layer

    sig = signature(func)

    for param_name, param in sig.parameters.items():
        # Skip parameters that are explicitely defined in the `parameters={}` field of the decorator
        if param_name in parameters:
            continue

        annotation = param.annotation
        default = param.default

        if annotation is _empty:
            # If the absence of type hints, we look at the type of eventual default values
            if default is _empty:
                # Last resort: is the parameter named unambiguously?
                if param_name in ["image", "img"]:
                    parameters[param_name] = skt.Image()
                elif param_name in ["mask", "msk", "segmentation", "seg"]:
                    parameters[param_name] = skt.Mask()
                elif param_name in ["points", "pts"]:
                    parameters[param_name] = skt.Points()
                elif param_name == "vectors":
                    parameters[param_name] = skt.Vectors()
                elif param_name == "tracks":
                    parameters[param_name] = skt.Tracks()
                elif param_name == "boxes":
                    parameters[param_name] = skt.Boxes()
                elif param_name == "paths":
                    parameters[param_name] = skt.Paths()
                else:
                    raise Exception(
                        f"Could not parse this parameter: {param_name}. Reason: No type hint or default provided."
                    )
            elif default is None:
                # Ignore parameters that default to None
                continue
            else:
                if isinstance(default, RECOGNIZED_TYPES):
                    layer = get_data_layer_type(type(default), default, param_name)
                    if layer:
                        parameters[param_name] = layer
                else:
                    raise Exception(
                        f"Could not parse this parameter: {param_name}. Reason: Parameter default suggests an unrecognized type."
                    )
        else:
            if annotation in RECOGNIZED_TYPES:
                layer = get_data_layer_type(annotation, default, param_name)
                if layer:
                    parameters[param_name] = layer
            else:
                raise Exception(
                    f"Could not parse this parameter: {param_name}. Reason: Parameter type hint suggests an unrecognized type."
                )

    return parameters


def _parse_pydantic_params_schema(
    run_algorithm_func: Callable, params_from_decorator: Dict
) -> BaseModel:
    """Convert the parameters dictionary provided by @algorithm_server to a Pydantic model."""
    # Parse the provided parameters dictionary + run function signature to a dict(str: DataLayer)
    parsed_params: Dict[str, DataLayer] = _parse_run_func_signature(
        run_algorithm_func, params_from_decorator
    )

    # Convert the dict(str: DataLayer) to a Pydantic BaseModel
    fields = {}
    validators = {}
    if parsed_params is not None:
        for param_name, layer in parsed_params.items():
            field_constraints = {"json_schema_extra": {}}
            if hasattr(layer, "min"):
                field_constraints["ge"] = layer.min
            if hasattr(layer, "max"):
                field_constraints["le"] = layer.max
            if hasattr(layer, "default"):
                field_constraints["default"] = layer.default
                field_constraints["json_schema_extra"]["example"] = layer.default
            else:
                if get_origin(layer.type) is Union and type(None) in get_args(
                    layer.type
                ):
                    field_constraints["default"] = None
            if hasattr(layer, "name"):
                field_constraints["title"] = layer.name
            if hasattr(layer, "description"):
                field_constraints["description"] = layer.description
            if hasattr(layer, "step"):
                field_constraints["json_schema_extra"]["step"] = layer.step
            if hasattr(layer, "auto_call"):
                field_constraints["json_schema_extra"]["auto_call"] = layer.auto_call

            field_constraints["json_schema_extra"]["param_type"] = layer.kind

            val_func = partial(
                DATA_TYPES.get(layer.kind)()._decode_and_validate,
                meta=layer.meta,
            )
            validators[f"validate_{param_name}"] = field_validator(
                param_name, mode="after"
            )(val_func)

            fields[param_name] = (layer.type, Field(**field_constraints))

    return create_model(
        "Parameters",
        __base__=Parameters,
        __validators__=validators,
        **fields,
    )


def _parse_user_func_output(payload: Any) -> Results:
    """Parse the user's function output to a Results object."""
    if payload is None:
        return None

    # payload => List[DataLayer]
    layers = []
    if isinstance(payload, (DataLayer, int, float, bool, str, np.ndarray, List, Tuple)):
        if isinstance(payload, DataLayer):
            layers.append(payload)
        else:
            if isinstance(payload, (List, Tuple)):
                for content in payload:
                    if isinstance(content, Tuple):
                        # (Legacy) Is it a result data tuple?
                        if (
                            (len(content) != 3)
                            | (not isinstance(content[1], dict))
                            | (not content[2] in DATA_TYPES.keys())
                        ):
                            raise Exception(
                                "Invalid algorithm return format: ", type(payload)
                            )
                        data, meta, kind = content
                        name = meta.pop("name")
                        layer_class = DATA_TYPES.get(kind)
                        layer = layer_class(data=data, name=name, meta=meta)
                        layers.append(layer)
                    else:
                        if isinstance(content, DataLayer):
                            layers.append(content)
                        elif isinstance(content, RECOGNIZED_TYPES):
                            layers.append(layer_class(TYPE_MAPPINGS[type(content)]))
                        else:
                            raise Exception(
                                "Invalid algorithm return format: ", type(content)
                            )
            else:
                layers.append(TYPE_MAPPINGS[type(payload)](payload))
    else:
        raise Exception("Invalid algorithm return format: ", type(payload))

    # List[DataLayer] => Results
    results = Results()
    for layer in layers:
        results.create(
            kind=layer.kind, data=layer.data, name=layer.name, meta=layer.meta
        )

    return results


class Algorithm(AlgorithmRunner):
    def __init__(
        self,
        run_algorithm_func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        title: str = "Image Processing Algorithm",
        description: str = "Implementation of an image processing algorithm.",
        tags: Optional[List[str]] = None,
        project_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
        metadata_file: str = "metadata.yaml",
        sample_images: Optional[List[Any]] = None,
    ):
        # Initialize mutables
        if tags is None:
            tags = []
        if sample_images is None:
            sample_images = []
        if parameters is None:
            parameters = {}

        # Resolve the algo name (if None => use algo function name)
        if name is None:
            name = run_algorithm_func.__name__
        self.name = name

        # Algorithm's run function from the user
        self._run_algorithm_func = run_algorithm_func
        update_wrapper(self, self._run_algorithm_func)  # improve function emulation

        # Sample images
        self.sample_images = sample_images

        # Resolve the Pydantic parameters model
        self.parameters_model = _parse_pydantic_params_schema(
            run_algorithm_func, parameters
        )

        # Initialize metadata info
        self.algo_info = etc.parse_algo_info(
            metadata_file, name, title, description, project_url, tags
        )

        self._algorithms = [name]

    @property
    def algorithms(self) -> Iterable[str]:
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms: Iterable[str]):
        self._algorithms = algorithms

    def __call__(self, *args, **kwargs):
        # Get a Results object
        results = self.run(*args, **kwargs)

        # Only return the data to emulate the wrapped function behavior
        to_return = [r.data for r in results]
        n_returns = len(to_return)
        if n_returns == 0:
            return
        elif n_returns == 1:
            return to_return[0]
        else:
            return to_return

    def __getattr__(self, name):
        """
        Algorithm attributes emulate function attributes
        (e.g. __doc__, __name__, __annotations__, __defaults__...)
        """
        return getattr(self._run_algorithm_func, name)

    @validate_algorithm
    def info(self, algorithm=None):
        """Create and open the algorithm info page in a web browser."""
        algo_params_schema = self.get_parameters(algorithm)
        etc.open_doc_link(algo_params_schema, algo_info=self.algo_info)

    @validate_algorithm
    def get_parameters(self, algorithm=None) -> dict:
        return self.parameters_model.model_json_schema()

    @validate_algorithm
    def get_sample_images(  # TODO: modify
        self, algorithm=None, first_only: bool = False
    ) -> Iterable[np.ndarray]:
        images = []
        if len(self.sample_images) > 0:
            for sample_image_or_path in self.sample_images:
                if isinstance(sample_image_or_path, np.ndarray):
                    images.append(sample_image_or_path)
                else:
                    images.append(skimage.io.imread(sample_image_or_path))
                if first_only:
                    images = images[0]
                    break
        return images

    @validate_algorithm
    def get_signature_params(self, algorithm=None) -> List[str]:
        """List parameter names of the algo run function."""
        return list(signature(self._run_algorithm_func).parameters.keys())

    def _is_stream(self, algorithm=None) -> bool:
        return isgeneratorfunction(self._run_algorithm_func)

    def _stream(self, algorithm, **algo_params):
        try:
            self.parameters_model(**algo_params)
        except ValidationError as e:
            raise e

        wrap = AlgoStream(self._run_algorithm_func(**algo_params))
        for payload in algo_stream_gen(wrap):
            yield _parse_user_func_output(payload)

    def _tile(
        self,
        algorithm,
        tile_size_px,
        overlap_percent,
        delay_sec,
        randomize,
        **algo_params,
    ):
        """Process the image sequentially in tiles."""
        algo_param_defs = self.get_parameters(algorithm).get("properties")
        for algo_params_tile, tile_info in etc.generate_tiles(
            algo_param_defs,
            algo_params,
            tile_size_px,
            overlap_percent,
            delay_sec,
            randomize,
        ):
            results = self._run(algorithm, **algo_params_tile)
            for layer in results:
                layer.meta = layer.meta | tile_info
            yield results
        return []

    def _run(self, algorithm, **algo_params) -> Iterable[Tuple]:
        try:
            self.parameters_model(**algo_params)
        except ValidationError as e:
            raise e

        payload = self._run_algorithm_func(**algo_params)
        return _parse_user_func_output(payload)


def algorithm(
    func: Optional[Callable] = None,
    parameters: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    title: str = "Image Processing Algorithm",
    description: str = "Implementation of an image processing algorithm.",
    tags: Optional[List[str]] = None,
    project_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
    metadata_file: str = "metadata.yaml",
    sample_images: Optional[List[Any]] = None,
):
    def _decorate(run_aglorithm_func: Callable):
        return Algorithm(
            run_algorithm_func=run_aglorithm_func,
            parameters=parameters,
            name=name,
            title=title,
            description=description,
            tags=tags,
            project_url=project_url,
            metadata_file=metadata_file,
            sample_images=sample_images,
        )

    if func is not None and callable(func):
        return _decorate(func)

    return _decorate
