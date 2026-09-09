from functools import partial, update_wrapper
from inspect import _empty, isgeneratorfunction, signature
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import skimage.io
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
)
from imaging_server_kit.core.errors import AlgorithmRuntimeError
import imaging_server_kit.core._etc as etc
import imaging_server_kit.types as skt
from imaging_server_kit.core.stack import Stack
from imaging_server_kit.core.runner import (
    AlgorithmRunner,
    validate_algorithm,
)
from imaging_server_kit.types import DATA_TYPES, Layer, layer_factory
from imaging_server_kit.validation.layer_validator import LayerValidator

TYPE_MAPPINGS: Dict[Any, Type[Layer]] = {
    int: skt.Integer,
    float: skt.Float,
    bool: skt.Bool,
    str: skt.String,
    np.ndarray: skt.Image,
    type(None): skt.Null,
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
    skt.Notification: skt.Notification,
}


### Parameters parsing ###


class Parameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _layer_from_type(hinted_type, default, param_name: str) -> Layer:
    if hinted_type not in TYPE_MAPPINGS:
        print(
            f"⚠️ Parameter `{param_name}` is of an unrecognized type "
            f"(`Hinted: {hinted_type}` ; Default: `{default}`). "
            "It will be treated as a `sk.Any` type and the default value will be ignored."
        )
        return skt.Any(name=param_name)

    cls: Type[Layer] = TYPE_MAPPINGS[hinted_type]
    return (
        cls(name=param_name)
        if default is _empty
        else cls(name=param_name, default=default)
    )


def _resolve_param_layer(param_name: str, annotation, default) -> Layer:
    """Resolve a parameter to its Layer type from a type hint annotation or a default value."""
    # First we check if a type hint was provided:
    if annotation is not _empty:
        return _layer_from_type(annotation, default, param_name)

    # If not, we check if there are recognizable defaults:
    if default is _empty:
        if param_name in DATA_TYPES:
            return DATA_TYPES[param_name]()
        return skt.Any(name=param_name)

    if isinstance(default, Layer):
        return default

    return _layer_from_type(type(default), default, param_name)


def _parse_run_func_signature(
    func: Callable, parameters: Dict[str, Layer]
) -> Dict[str, Layer]:
    resolved = dict(parameters)

    for param_name, param in signature(func).parameters.items():
        if param_name in resolved:
            if not isinstance(resolved[param_name], Layer):
                raise TypeError(f"Parameter '{param_name}' should be a Layer instance.")
            continue
        resolved[param_name] = _resolve_param_layer(
            param_name, param.annotation, param.default
        )

    return resolved


def _field_constraints_from_layer(layer: Layer) -> dict:
    meta = layer.meta or {}
    constraints = {}
    if "default" in meta:
        constraints["default"] = meta["default"]
    if "min" in meta:
        constraints["ge"] = meta["min"]
    if "max" in meta:
        constraints["le"] = meta["max"]

    return {
        "title": layer.name,
        "description": meta.get("description"),
        "json_schema_extra": {"param_type": layer.kind} | meta,
        **constraints,
    }


def _parse_pydantic_params_schema(
    run_algorithm_func: Callable, params_from_decorator: Dict
):
    parsed_params = _parse_run_func_signature(run_algorithm_func, params_from_decorator)

    layer_validator = LayerValidator()  # stateless — hoist out of the loop
    fields, validators = {}, {}

    for param_name, layer in parsed_params.items():
        validators[f"validate_{param_name}"] = field_validator(
            param_name, mode="after"
        )(partial(layer_validator.validate, layer=layer))
        fields[param_name] = (layer.type, Field(**_field_constraints_from_layer(layer)))

    return create_model(
        "Parameters", __base__=Parameters, __validators__=validators, **fields
    )


### Function output parsing ###


def _parse_output(payload: Any) -> Layer:
    if isinstance(payload, Layer):
        return payload
    if isinstance(payload, tuple(TYPE_MAPPINGS.keys())):  # Not a data layer...
        cls: Type[Layer] = TYPE_MAPPINGS[type(payload)]
        return cls(data=payload)
    else:
        # Unidentified return types are wrapped as the `Any` type:
        return skt.Any(data=payload)


def _parse_payload(payload: Any) -> Union[List[Layer], Layer]:
    if isinstance(payload, (List, Tuple)):  # Multiple returns
        return [_parse_payload(p) for p in payload]  # type: ignore
    else:
        return _parse_output(payload)


def _parse_user_func_output(payload: Any) -> Stack:
    """Parse the user's function output to a Stack object."""
    # payload => List[Layer]
    layers = _parse_payload(payload)
    if not isinstance(layers, List):
        layers = [layers]

    # List[Layer] => Stack
    return Stack(layers=layers)


### AlgoStream utility ###


class AlgoStream:
    def __init__(self, gen):
        self._it = iter(gen)
        self.value = None

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except (StopIteration, AlgorithmRuntimeError) as e:
            if isinstance(e, StopIteration):
                self.value = e.value
                raise
            elif isinstance(e, AlgorithmRuntimeError):
                raise e


def algo_stream_gen(algo_stream: AlgoStream) -> Generator[Any, None, None]:
    for x in algo_stream:
        yield x
    if algo_stream.value is not None:
        yield algo_stream.value


### Algorithm implementation ###


class Algorithm(AlgorithmRunner):
    """An algorithm built by wrapping a Python function. Usually created via the `@sk.algorithm(...)` decorator rather than instantiated directly.

    Parameters
    ----------
    run_algorithm_func: The Python function to convert.
    parameters: A dictionary of annotated parameters.
    name: A name for the algorithm (doesn't accept spaces and special characters).
    description: A short description to display on the algorithm doc page.
    tags: A list of tags (arbitrary).
    project_url: A link to a related, or the original project (gets displayed on the algo doc page).
    metadata_file: A path to a metadata.yaml file with algorithm metadata.
    samples: A list of sample parameters for the algorithm, each represented as a dictionary mapping parameter_name to example_value. Sample images can be a Numpy array, a URL, or a local path to a file readable by `skimage.io.imread`.
    tileable: Whether to allow running the algorithm tile-by-tile.

    Notes
    ----------
    - Algorithms can be converted to FastAPI servers or PyQt widgets for Napari or QuPath.
    - Algorithms are associated with a Pydantic Schema that can be used to validate input parameters.
    - Algorithms can be run tile-by-tile (in most cases).
    - Algorithms can be run on a subset of the spatial domain defined by their inputs.
    - Algorithms can provide `samples` (example inputs).
    - Algorithm metadata can populate an `info` page.

    Attributes
    ----------
    name: A name for the algorithm.
    parameters_model: A JSON schema representation of algorithm parameters.
    samples: A list of sample parameters, mapping parameter names to parameter values.
    algo_info: A dictionary of metadata about the algorithm.
    algorithms: A list containing the algorithm's name.

    Methods
    ----------
    run(): Execute the algorithm with a set of parameters.
        Set `tiled=True` for tiled inference.
        Raises a ValidationError when parameters are invalidated.
    get_n_samples(): Get the number of samples available.
    get_sample(): Get a sample by index.
    info(): Access algorithm documentation.
    get_parameters(): Get the algorithm parameters schema.
    """

    def __init__(
        self,
        run_algorithm_func: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        description: str = "Implementation of an image processing algorithm.",
        tags: Optional[List[str]] = None,
        project_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
        metadata_file: str = "metadata.yaml",
        samples: Optional[List[Dict[str, Any]]] = None,
        tileable: bool = False,
    ):
        # Initialize mutables
        if tags is None:
            tags = []
        if samples is None:
            samples = []
        if parameters is None:
            parameters = {}

        # Resolve the algo name (if None => use algo function name)
        if name is None:
            name = run_algorithm_func.__name__
        self._name = name

        # Algorithm's run function from the user
        self._run_algorithm_func = run_algorithm_func
        update_wrapper(self, self._run_algorithm_func)  # improve function emulation

        # Samples
        self.samples = samples

        # Tileability
        self._tileable = tileable

        # Resolve the Pydantic parameters model
        self.parameters_model = _parse_pydantic_params_schema(
            run_algorithm_func, parameters
        )

        # Initialize metadata info
        self.algo_info = etc.parse_algo_info(
            metadata_file, self._name, description, project_url, tags
        )

        self._algorithms = [self._name]

    @property
    def name(self) -> str:
        return self._name

    @property
    def algorithms(self) -> Union[List[str], Tuple[str, ...]]:
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms: Union[List[str], Tuple[str, ...]]):
        self._algorithms = algorithms

    @property
    def tileable(self) -> bool:
        return self._tileable

    @tileable.setter
    def tileable(self, tileable: bool):
        self._tileable = tileable

    def __call__(self, *args, **kwargs) -> Any:
        # Get a Stack object
        stack = self.run(*args, **kwargs)

        # Only return the data to emulate the wrapped function behavior
        to_return = [l.data for l in stack]
        n_returns = len(to_return)
        if n_returns == 0:
            return
        elif n_returns == 1:
            return to_return[0]
        else:
            return to_return

    def __str__(self):
        return f"{self.name} (algorithm)"

    def __getattr__(self, name):
        """
        Algorithm attributes emulate function attributes
        (e.g. __doc__, __name__, __annotations__, __defaults__...)
        """
        return getattr(self._run_algorithm_func, name)

    @validate_algorithm
    def info(self, algorithm: Optional[str] = None) -> None:
        """Create and open the algorithm info page in a web browser."""
        algo_params_schema = self.get_parameters(algorithm)
        etc.open_doc_link(algo_params_schema, algo_info=self.algo_info)

    @validate_algorithm
    def get_parameters(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        return self.parameters_model.model_json_schema()

    @validate_algorithm
    def get_sample(
        self, algorithm: Optional[str] = None, idx: int = 0
    ) -> Optional[Stack]:
        n_samples = self.get_n_samples(algorithm)
        if n_samples == 0:
            return

        if idx > n_samples - 1:
            raise ValueError(
                f"Algorithm provides {n_samples} samples. Max value for `idx` is {n_samples-1}!"
            )

        algo_params_defs = self.get_parameters(algorithm)["properties"]
        signature_params = self.get_signature_params(algorithm)
        resolved_params = etc.resolve_params(
            algo_param_defs=algo_params_defs,
            signature_params=signature_params,
            args=(),
            algo_params=self.samples[idx],
        )
        # Convert the sample to a Stack object
        sample_stack = Stack()
        for param_name, param_value in resolved_params.items():
            kind = algo_params_defs.get(param_name).get("param_type")
            if (kind in ["image", "mask"]) & (not isinstance(param_value, np.ndarray)):
                param_value = skimage.io.imread(param_value)

            # Set Min/Max contrast limits for images, by default
            kw = {}
            if kind == "image":
                kw["contrast_limits"] = [
                    float(param_value.min()),
                    float(param_value.max()),
                ]

            layer = layer_factory(kind=kind, data=param_value, name=param_name, **kw)
            sample_stack.add(layer)

        return sample_stack

    def get_n_samples(self, algorithm: Optional[str] = None) -> int:
        return len(self.samples)

    def is_tileable(self, algorithm: Optional[str] = None) -> bool:
        return self.tileable

    @validate_algorithm
    def get_signature_params(self, algorithm: Optional[str] = None) -> List[str]:
        """List parameter names of the algo run function."""
        return list(signature(self._run_algorithm_func).parameters.keys())

    def _stream(
        self, algorithm: str, params_stack: Stack
    ) -> Generator[Stack, None, None]:
        """Generator that runs an algorithm using given parameters."""
        algo_params = {l.name: l.data for l in params_stack.layers}

        # Validate parameters `manually` with Pydantic:
        try:
            self.parameters_model(**algo_params)
        except ValidationError as e:
            raise e

        # If user-defined run function has `yield` statements:
        if isgeneratorfunction(self._run_algorithm_func):
            gen = algo_stream_gen(AlgoStream(self._run_algorithm_func(**algo_params)))
            try:
                for payload in gen:
                    yield _parse_user_func_output(payload)
            except AlgorithmRuntimeError:
                raise
            except Exception as e:
                raise AlgorithmRuntimeError(algorithm=algorithm, error=e)
        # Otherwise:
        else:
            try:
                payload = self._run_algorithm_func(**algo_params)
                yield _parse_user_func_output(payload)
            except Exception as e:
                raise AlgorithmRuntimeError(algorithm=algorithm, error=e)


def algorithm(
    func: Optional[Callable] = None,
    parameters: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    description: str = "Implementation of an image processing algorithm.",
    tags: Optional[List[str]] = None,
    project_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
    metadata_file: str = "metadata.yaml",
    samples: Optional[List[Dict[str, Any]]] = None,
    tileable: bool = False,
) -> Union[Algorithm, Callable]:
    """Wrap a Python function as an algorithm (sk.Algorithm). Typically used as the `@sk.algorithm(...)` decorator.

    Parameters
    ----------
    func : The Python function to convert.
    parameters : A dictionary of annotated parameters.
    name: A name for the algorithm (doesn't accept spaces and special characters).
    description: A short description to display on the algorithm doc page.
    tags: A list of tags (arbitrary).
    project_url: A link to a related, or the original project (gets displayed on the algo doc page).
    metadata_file: A path to a metadata.yaml file with algorithm metadata.
    samples: A list of sample parameters for the algorithm, each represented as a dictionary mapping parameter_name to example_value. Sample images can be a Numpy array, a URL, or a local path to a file readable by `skimag.io.imread`.
    tileable: Whether to allow running the algorithm tile-by-tile.

    Returns
    -------
    An algorithm instance (sk.Algorithm).
    """

    def _decorate(run_aglorithm_func: Callable) -> Algorithm:
        return Algorithm(
            run_algorithm_func=run_aglorithm_func,
            parameters=parameters,
            name=name,
            description=description,
            tags=tags,
            project_url=project_url,
            metadata_file=metadata_file,
            samples=samples,
            tileable=tileable,
        )

    if func is not None and callable(func):
        return _decorate(func)

    return _decorate
