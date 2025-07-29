import asyncio
import importlib.resources
import os
from contextlib import asynccontextmanager
from inspect import isgeneratorfunction
from pathlib import Path
from typing import List, Tuple, Type

import msgpack
import numpy as np
import requests
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict

from imaging_server_kit._version import __version__
from imaging_server_kit.core.encoding import encode_contents
from imaging_server_kit.core.serialization import serialize_result_tuple

templates_dir = Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("templates")
)
static_dir = Path(
    importlib.resources.files("imaging_server_kit.core").joinpath("static")
)
templates = Jinja2Templates(directory=str(templates_dir))

PROCESS_TIMEOUT_SEC = 3600  # Client timeout for the /process route


def load_from_yaml(file_path: str):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def parse_algo_params_schema(algo_params_schema):
    algo_params = algo_params_schema.get("properties")
    required_params = algo_params_schema.get("required")
    for param in algo_params.keys():
        if required_params is None:
            algo_params[param]["required"] = False
        else:
            algo_params[param]["required"] = param in required_params
    return algo_params


ALGORITHM_HUB_URL = os.getenv("ALGORITHM_HUB_URL", "http://algorithm_hub:8000")


class Parameters(BaseModel):
    ...

    model_config = ConfigDict(extra="forbid")


class AlgorithmServer:
    def __init__(
        self,
        parameters_model: Type[BaseModel] = Parameters,
        algorithm_name: str = "algorithm",
        metadata_file: str = "metadata.yaml",
        title: str = "Image Processing Algorithm",
        description: str = "Implementation of an image processing algorithm.",
        tags: List[str] = [],
        used_for: List[str] = [],
        project_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
        serverkit_repo_url: str = "https://github.com/Imaging-Server-Kit/imaging-server-kit",
    ):
        self.algorithm_name = algorithm_name
        self.parameters_model = parameters_model

        if metadata_file is not None and Path(metadata_file).exists():
            self.algo_info = load_from_yaml(metadata_file)
        else:
            self.algo_info = {
                "project_slug": algorithm_name,
                "project_url": project_url,
                "serverkit_repo_url": serverkit_repo_url,
                "project_name": title,
                "description": description,
                "used_for": used_for,
                "tags": tags,
            }

        self.app = FastAPI(title=algorithm_name, lifespan=self.lifespan)

        # Centralized exception handlers
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(
            request: Request, exc: RequestValidationError
        ):
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"error_type": "validation_error", "detail": exc.errors()},
            )

        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            # Let FastAPI handle HTTPExceptions
            if isinstance(exc, HTTPException):
                raise exc
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error_type": "internal_server_error", "message": str(exc)},
            )

        # Info HTML
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        self.register_routes()

        self.services = [self.algorithm_name]

    @property
    def _is_stream(self):
        return isgeneratorfunction(self.run_algorithm)
    
    @property
    def parameters(self):
        return self.parameters_model.model_json_schema()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        await self.register_with_algohub()
        yield
        await self.deregister_from_algohub()

    async def register_with_algohub(self):
        try:
            response = requests.get(f"{ALGORITHM_HUB_URL}/")
        except Exception:
            print("Algorithm hub unavailable.")
            return

        response = requests.post(
            f"{ALGORITHM_HUB_URL}/register",
            json={
                "name": self.algorithm_name,
                "url": f"http://{self.algorithm_name}:8000",
            },
        )
        if response.status_code == 201:
            print(f"Service {self.algorithm_name} registered successfully.")
        else:
            print(f"Failed to register {self.algorithm_name}: {response.json()}")

    async def deregister_from_algohub(self):
        deregister_url = f"{ALGORITHM_HUB_URL}/deregister"
        response = requests.post(deregister_url, json={"name": self.algorithm_name})
        if response.status_code == 201:
            print(f"Service {self.algorithm_name} deregistered.")
        else:
            print(f"Failed to deregister {self.algorithm_name}: {response.json()}")

    def register_routes(self):
        @self.app.get(
            "/",
            summary="Home or info page",
            description="Serve the algorithm info page (as home page).",
            tags=["meta"],
        )
        def home(request: Request):
            return info(request)

        @self.app.get(
            "/services",
            response_model=dict,
            summary="List services",
            description="Return the list of available algorithm services.",
            tags=["meta"],
        )
        def list_services():
            """List available algorithm services."""
            return {"services": self.services}

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
            f"/{self.algorithm_name}/info",
            response_class=HTMLResponse,
            summary="Algorithm info",
            description="Render the HTML info page for this algorithm.",
            tags=["algorithm", "meta"],
        )
        def info(request: Request):
            algo_params_schema = get_algo_params()
            algo_params = parse_algo_params_schema(algo_params_schema)
            return templates.TemplateResponse(
                "info.html",
                {
                    "request": request,
                    "algo_info": self.algo_info,
                    "algo_params": algo_params,
                },
            )

        @self.app.get(
            f"/{self.algorithm_name}/is_stream",
            description="Whether to handle the algorithm as a stream.",
            tags=["algorithm", "meta"],
        )
        def is_stream():
            return self._is_stream

        @self.app.post(
            f"/{self.algorithm_name}/process",
            status_code=status.HTTP_201_CREATED,
            summary="Run algorithm",
            description="Execute the algorithm with the provided parameters and return the serialized results.",
            tags=["algorithm"],
        )
        async def run_algo(
            algo_params: self.parameters_model,
        ):
            """Run the algorithm processing endpoint."""
            return await self._run_algo_logic(algo_params)

        @self.app.post(
            f"/{self.algorithm_name}/stream",
            status_code=status.HTTP_200_OK,
            summary="Stream algorithm",
            description="Execute the algorithm as a stream with the provided parameters.",
            tags=["algorithm"],
        )
        async def stream_algo(
            algo_params: self.parameters_model,
        ):
            """Run the algorithm as a stream."""
            return StreamingResponse(
                content=self._stream_algo_results(algo_params),
                media_type="application/msgpack",
            )

        @self.app.get(
            f"/{self.algorithm_name}/parameters",
            response_model=dict,
            summary="Get parameter schema",
            description="Return the JSON schema for this algorithm's parameters.",
            tags=["algorithm"],
        )
        def get_algo_params():
            """Get the parameter JSON schema."""
            return self.parameters

        @self.app.get(
            f"/{self.algorithm_name}/sample_images",
            response_model=dict,
            summary="Get sample images",
            description="Return a list of encoded sample images for this algorithm.",
            tags=["algorithm"],
        )
        def get_sample_images():
            """Fetch and encode sample images."""
            images = self.load_sample_images()
            encoded_images = [
                {"sample_image": encode_contents(image)} for image in images
            ]
            return {"sample_images": encoded_images}

    def _stream_algo_results(self, algo_params):
        """Generator that yields the (serialized) results of run_algorithm with the provided parameters."""
        for data_tuple_list in self.run_algorithm(**algo_params.dict()):
            serialized_results = serialize_result_tuple(data_tuple_list)
            for r in serialized_results:
                yield msgpack.packb(r)

    async def _serialize_result_tuple(self, result_data_tuple):
        serialized_results = await asyncio.to_thread(
            serialize_result_tuple, result_data_tuple
        )
        return serialized_results

    async def _run_algo_logic(self, algo_params):
        try:
            result_data_tuple = await asyncio.wait_for(
                self._run_algorithm(
                    **algo_params.dict()
                ),  # TODO: Pydantic warns: ` The `dict` method is deprecated; use `model_dump` instead.`
                timeout=PROCESS_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:  # TODO: This doesn't kill the thread...
            raise HTTPException(
                status_code=504, detail="Request timeout during processing."
            )
        try:
            serialized_results = await asyncio.wait_for(
                self._serialize_result_tuple(result_data_tuple),
                timeout=PROCESS_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504, detail="Request timeout during serialization."
            )

        # TODO: would it make sense to use msgpack here too?
        # serialized_results = [msgpack.packb(r) for r in serialized_results]  # This doesn't work...

        return serialized_results

    async def _run_algorithm(self, **algo_params):
        result_data_tuple = await asyncio.to_thread(self.run_algorithm, **algo_params)
        return result_data_tuple

    def load_sample_images(self) -> List["np.ndarray"]:
        raise NotImplementedError("Subclasses should implement this method")

    def run_algorithm(self, **algo_params) -> List[Tuple]:
        raise NotImplementedError("Subclasses should implement this method")
