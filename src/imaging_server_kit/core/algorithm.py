from functools import partial, update_wrapper
from inspect import _empty, isgeneratorfunction, signature
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
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
from imaging_server_kit.core.results import Results
from imaging_server_kit.core.runner import (
    AlgorithmRunner,
    AlgoStream,
    algo_stream_gen,
    validate_algorithm,
)
from imaging_server_kit.types import DATA_TYPES, DataLayer

TYPE_MAPPINGS: Dict[Any, Type[DataLayer]] = {
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
    # skt.Choice: skt.Choice,  # Won't work
    skt.Notification: skt.Notification,
}

RECOGNIZED_TYPES = tuple(TYPE_MAPPINGS.keys())

### Parameters parsing ###


class Parameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


def _parse_run_func_signature(
    func: Callable,
    parameters: Dict[str, DataLayer],
) -> Dict[str, DataLayer]:
    """Resolve parameters into {param_name : DataLayer} based on the annotations from the decorator
    and the signature of the wrapped Python function."""

    def get_data_layer_type(hinted_type, default, param_name) -> DataLayer:
        cls: Type[DataLayer] = TYPE_MAPPINGS[hinted_type]
        if default is _empty:
            return cls(name=param_name)
        else:
            return cls(name=param_name, default=default) # type: ignore

    resolved = dict(
        parameters
    )  # Copy of the original parameters to avoid mutating them

    sig = signature(func)

    for param_name, param in sig.parameters.items():
        # Skip parameters explicitely defined in `parameters={}` (check that they are DataLayer instances)
        if param_name in resolved:
            if not isinstance(resolved.get(param_name), DataLayer):
                raise TypeError(
                    f"Parameter '{param_name}' should be a DataLayer instance."
                )
            continue

        annotation = param.annotation  # Type hints
        default = param.default

        # Check type hints
        if annotation is _empty:
            # If the absence of type hints, we look at the type of eventual default values
            if default is _empty:
                # Last resort: is the parameter named unambiguously (variable name == parameter `kind`, for example the variable is named `image`)?
                if param_name in DATA_TYPES:
                    cls: Type[DataLayer] = DATA_TYPES[param_name]
                    resolved[param_name] = cls()  # Initialized with defaults
                else:
                    raise TypeError(
                        f"Could not parse this parameter: '{param_name}'. Reason: No type hint or default provided."
                    )
            else:
                if isinstance(default, RECOGNIZED_TYPES):
                    if isinstance(default, DataLayer):
                        # Case where default is a data layer, for example user has defaulted x=sk.Float(...)
                        resolved[param_name] = default
                    else:
                        # int, float, str, None defaults...
                        default_type = type(default)
                        resolved[param_name] = get_data_layer_type(
                            default_type, default, param_name
                        )
                else:
                    raise TypeError(
                        f"Could not parse this parameter: '{param_name}'. Reason: Parameter default is an unrecognized type."
                    )
        else:
            if annotation in RECOGNIZED_TYPES:
                resolved[param_name] = get_data_layer_type(
                    annotation, default, param_name
                )
            else:
                raise TypeError(
                    f"Could not parse this parameter: '{param_name}'. Reason: Parameter type hint is an unrecognized type."
                )

    return resolved


def _parse_pydantic_params_schema(
    run_algorithm_func: Callable,
    params_from_decorator: Dict,
):
    """Convert the parameters dictionary provided by @algorithm_server to a Pydantic model."""
    # Parse the provided parameters dictionary + run function signature to a dict(str: DataLayer)
    parsed_params: Dict[str, DataLayer] = _parse_run_func_signature(
        run_algorithm_func, params_from_decorator
    )

    # Generate a Pydantic BaseModel
    fields = {}
    validators = {}
    for param_name, data_layer in parsed_params.items():
        constraints = data_layer.constraints
        main, extra = constraints
        field_constraints = {
            "title": data_layer.name,
            "description": data_layer.description,
            "json_schema_extra": {"param_type": data_layer.kind} | extra,
        } | main

        # Resolve the validator function
        val_func = partial(
            data_layer._validate, meta=data_layer.meta, constraints=constraints
        )
        validators[f"validate_{param_name}"] = field_validator(
            param_name, mode="after"
        )(val_func)

        fields[param_name] = (data_layer.type, Field(**field_constraints))

    return create_model(
        "Parameters",
        __base__=Parameters,
        __validators__=validators,
        **fields,
    )


### Function output parsing ###

def _parse_output(payload: Any) -> DataLayer:
    if isinstance(payload, DataLayer):
        return payload
    if isinstance(payload, RECOGNIZED_TYPES):  # Not a datalayer...
        cls: Type[DataLayer] = TYPE_MAPPINGS[type(payload)]
        return cls(data=payload)
    else:
        raise TypeError(f"Function should return: List[DataLayer]. Got: {type(payload)}")


def _parse_payload(payload: Any) -> Union[List[DataLayer], DataLayer]:
    if isinstance(payload, (List, Tuple)):  # Multiple returns
        return [_parse_payload(p) for p in payload] # type: ignore
    else:
        return _parse_output(payload)


def _parse_user_func_output(payload: Any) -> Results:
    """Parse the user's function output to a Results object."""
    # payload => List[DataLayer]
    layers = _parse_payload(payload)
    if not isinstance(layers, List):
        layers = [layers]

    # List[DataLayer] => Results
    results = Results(layers)
    # for l in layers:
    #     results.create(l.kind, l.data, l.name, l.meta)
    return results


class Algorithm(AlgorithmRunner):
    """An algorithm converted from a Python function.

    Attributes
    ----------
    name: A name for the algorithm.
    samples: A list of sample parameters, mapping parameter names to parameter values.
    algo_info: A dictionary of metadata about the algorithm.
    parameters_model: A JSON schema representation of algorithm parameters.

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
        tileable: bool = True,
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
        self.name = name

        # Algorithm's run function from the user
        self._run_algorithm_func = run_algorithm_func
        update_wrapper(self, self._run_algorithm_func)  # improve function emulation

        # Samples
        self.samples = samples
        
        # Tileability
        self.tileable = tileable

        # Resolve the Pydantic parameters model
        self.parameters_model = _parse_pydantic_params_schema(
            run_algorithm_func, parameters
        )

        # Initialize metadata info
        self.algo_info = etc.parse_algo_info(
            metadata_file, name, description, project_url, tags
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

    def __str__(self):
        return f"{self.name} (algorithm)"

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
    def get_parameters(self, algorithm=None) -> Dict[str, Any]:
        return self.parameters_model.model_json_schema()

    @validate_algorithm
    def get_sample(self, algorithm=None, idx: int = 0) -> Optional[Results]:
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

        # Convert the sample to a Results object
        sample_results = Results()
        for param_name, param_value in resolved_params.items():
            kind = algo_params_defs.get(param_name).get("param_type")
            if (kind in ["image", "mask"]) & (not isinstance(param_value, np.ndarray)):
                param_value = skimage.io.imread(param_value)
            sample_results.create(kind, param_value, param_name)
        return sample_results

    def get_n_samples(self, algorithm=None):
        return len(self.samples)

    def is_tileable(self, algorithm=None):
        return self.tileable
    
    @validate_algorithm
    def get_signature_params(self, algorithm=None) -> List[str]:
        """List parameter names of the algo run function."""
        return list(signature(self._run_algorithm_func).parameters.keys())

    def _is_stream(self, algorithm=None) -> bool:
        return isgeneratorfunction(self._run_algorithm_func)

    def _stream(self, algorithm, param_results: Results):
        algo_params = param_results.to_params_dict()
        try:
            self.parameters_model(**algo_params)
        except ValidationError as e:
            raise e

        for payload in algo_stream_gen(
            AlgoStream(self._run_algorithm_func(**algo_params))
        ):
            yield _parse_user_func_output(payload)

    def _tile(
        self,
        algorithm: str,
        tile_size_px: int,
        overlap_percent: float,
        delay_sec: float,
        randomize: bool,
        param_results: Results,
    ):
        """Process the image sequentially in tiles."""
        for tile_results, tile_info in etc.generate_tiles(
            param_results,
            tile_size_px,
            overlap_percent,
            delay_sec,
            randomize,
        ):
            results = self._run(algorithm, tile_results)
            if results is not None:
                for layer in results:
                    layer.meta = layer.meta | tile_info
                yield results

    def _run(self, algorithm, param_results: Results) -> Results:
        algo_params = param_results.to_params_dict()
        try:
            self.parameters_model(**algo_params)
        except ValidationError as e:
            raise e
        try:
            payload = self._run_algorithm_func(**algo_params)
        except Exception as e:
            raise AlgorithmRuntimeError(algorithm, e)
        return _parse_user_func_output(payload)


def algorithm(
    func: Optional[Callable] = None,
    parameters: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    description: str = "Implementation of an image processing algorithm.",
    tags: Optional[List[str]] = None,
    project_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
    metadata_file: str = "metadata.yaml",
    samples: Optional[List[Dict[str, Any]]] = None,
    tileable: bool = True,
) -> Algorithm:
    """Convert a Python function into an algorithm instance (sk.Algorithm).

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
    tileable: Whether to allow running the algorithm tile-by-tile. Can be set to False to explicitely disable that functionality.
    
    Returns
    -------
    An algorithm instance (sk.Algorithm).
    """

    def _decorate(run_aglorithm_func: Callable):
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
