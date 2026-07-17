import importlib.resources
import os
import pathlib
from typing import Dict, Iterable, List, Optional

import msgpack
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

import imaging_server_kit.core._etc as etc
from imaging_server_kit._version import __version__
from imaging_server_kit.core.algorithm import Algorithm
from imaging_server_kit.core.stack import Stack
from imaging_server_kit.types import Layer, layer_factory
from imaging_server_kit.remote.stack_serializer import StackSerializer


templates_dir = pathlib.Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("templates")  # type: ignore
)

static_dir = pathlib.Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("static")  # type: ignore
)

templates = Jinja2Templates(directory=str(templates_dir))

ALGORITHM_HUB_URL = os.getenv("ALGORITHM_HUB_URL", "http://algorithm_hub:8000")


def find_algorithm(algorithm_name: str, algorithms_dict: Dict) -> Algorithm:
    if algorithm_name not in algorithms_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Algorithm {algorithm_name} not found",
        )
    algorithm = algorithms_dict[algorithm_name]
    return algorithm


class AlgorithmApp:
    def __init__(self, algorithms: List[Algorithm], name: str):
        self.algorithms_dict = {algo.name: algo for algo in algorithms}
        self.app = FastAPI(title=name)
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        # Centralized exception handlers
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"error_type": "validation_error", "detail": exc.errors()},
            )

        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            if isinstance(exc, HTTPException):
                raise exc
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error_type": "internal_server_error", "message": str(exc)},
            )

        self._register_routes()

    @property
    def algorithms(self) -> List[str]:
        return list(self.algorithms_dict.keys())

    def serve(self, host="0.0.0.0", port=8000, reload=False):
        """Run the algorithm server with uvicorn on the specified port for access on http://localhost:<port>."""
        uvicorn.run(self.app, host=host, port=port, reload=reload)

    def _register_routes(self):
        @self.app.get(
            "/",
            summary="Home or info page",
            description="Serve the algorithm info page (as home page).",
            tags=["meta"],
            response_class=HTMLResponse,
        )
        async def home(request: Request):
            if len(self.algorithms) > 1:
                return templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "algorithms": self.algorithms,
                    },
                )
            elif len(self.algorithms) == 1:
                return info(algorithm_name=self.algorithms[0], request=request)

        @self.app.get(
            "/algorithms",
            response_model=dict,
            summary="List algorithms",
            description="Return the list of available algorithm services.",
            tags=["meta"],
        )
        def list_algorithms():
            """List available algorithm services."""
            return {"algorithms": self.algorithms}

        @self.app.get(
            "/version",
            response_model=str,
            summary="Get server version",
            description="Return the version of the Imaging Server Kit.",
            tags=["meta"],
        )
        def get_version():
            """Get the package version."""
            return __version__

        @self.app.get(
            "/{algorithm_name}/info",
            response_class=HTMLResponse,
            summary="Algorithm info",
            description="Render the HTML info page for this algorithm.",
            tags=["algorithm"],
        )
        def info(algorithm_name: str = Path(...), request: Request = ...):
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            algo_info = algorithm.algo_info
            algo_params_schema = algorithm.get_parameters(algorithm_name)
            algo_params = etc.parse_algo_params_schema(algo_params_schema)
            return templates.TemplateResponse(
                request=request,
                name="info.html",
                context={
                    "algo_info": algo_info,
                    "algo_params": algo_params,
                },
            )

        @self.app.get(
            "/{algorithm_name}/parameters",
            response_model=dict,
            summary="Get parameter schema",
            description="Return the JSON schema for this algorithm's parameters.",
            tags=["algorithm"],
        )
        def get_algo_params(algorithm_name: str):
            """Get the parameter JSON schema."""
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            return algorithm.get_parameters(algorithm=algorithm_name)

        @self.app.get(
            "/{algorithm_name}/sample/{idx}",
            summary="Get sample parameters",
            description="Return encoded sample parameters for this algorithm.",
            tags=["algorithm"],
        )
        def get_sample(algorithm_name: str, idx: int):
            """Fetch and encode sample parameters."""
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            sample = algorithm.get_sample(algorithm=algorithm_name, idx=idx)
            if sample is not None:
                stack_serializer = StackSerializer()
                return stack_serializer.serialize(sample, "Python/Napari")

        @self.app.get(
            "/{algorithm_name}/n_samples",
            response_model=dict,
            summary="Get the number of samples availbale.",
            description="Return the number of samples available for this algorithm.",
            tags=["algorithm"],
        )
        def get_n_samples(algorithm_name: str):
            """Get the number of samples availbale."""
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            n_samples = algorithm.get_n_samples(algorithm=algorithm_name)
            return {"n_samples": n_samples}

        @self.app.get(
            "/{algorithm_name}/tileable",
            response_model=dict,
            summary="Whether the algorithm is tileable.",
            tags=["algorithm"],
        )
        def is_tileable(algorithm_name: str):
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            tileable = algorithm.is_tileable(algorithm=algorithm_name)
            return {"tileable": tileable}

        @self.app.get("/{algorithm_name}/signature", tags=["algorithm"])
        def get_signature(algorithm_name: str):
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            return algorithm.get_signature_params(algorithm_name)

        @self.app.post(
            "/{algorithm_name}/process",
            status_code=status.HTTP_200_OK,
            summary="Run algorithm",
            description="Execute the algorithm with the provided parameters.",
            tags=["algorithm"],
        )
        async def stream_algo(
            algorithm_name: str = Path(...),
            request: Request = ...,
        ):
            """Run the algorithm with the provided parameters."""
            algorithm = find_algorithm(algorithm_name, self.algorithms_dict)
            encoded_params = await request.json()

            # Python/Napari or Java/QuPath
            client_origin = str(request.headers.get("User-Agent"))

            # Reconstruct the algo parameters as a `Stack` object
            stack_serializer = StackSerializer()
            params_stack = stack_serializer.deserialize(encoded_params, client_origin)

            # Special case: when request is sent from QuPath, the image is named `qupath-image`
            # and should be assigned to whichever parameter is an image in the algo (we assume)
            # TODO: shouldn't this decision be handled by the QuPath extension?
            if client_origin == "Java/QuPath":
                qupath_image: Optional[Layer] = params_stack.read("image-qupath")
                if qupath_image is not None:
                    params_stack.delete("image-qupath")
                    # Find the first image parameter in the schema and assume it's what the QuPath image is meant to be
                    for param_name, param_values in algorithm.get_parameters()[
                        "properties"
                    ].items():
                        param_type = param_values.get("param_type")
                        if param_type == "image":
                            param_layer = layer_factory(
                                kind="image", data=qupath_image.data, name=param_name
                            )
                            params_stack.add(param_layer)
                            break

            # Validate the parameters `manually` with Pydantic...
            try:
                algo_params = {l.name: l.data for l in params_stack.layers}
                algorithm.parameters_model(**algo_params)
            except ValidationError as e:
                raise HTTPException(status_code=422, detail=e.errors())

            # Create the algorithm `run` generator
            gen = algorithm.run_generator(
                algorithm=algorithm_name,
                params_stack=params_stack,
            )

            # Will do stack.serialize() => msgpack.packrb() to stream the response
            stream = self._stream_msgpack(gen, client_origin)

            # To check: `content` is a Python generator, where StreamingResponse expects a special ContentStream object..
            # but it iseems to work anyway
            return StreamingResponse(stream, media_type="application/msgpack")

    def _stream_msgpack(self, stream_generator: Iterable[Stack], client_origin: str):
        stack_serializer = StackSerializer()
        for result_tile, params_tile in stream_generator:
            for r in stack_serializer.serialize(result_tile, client_origin):
                yield msgpack.packb(r)
