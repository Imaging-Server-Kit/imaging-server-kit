from typing import List
import os
import requests
from fastapi import FastAPI, Request, status, HTTPException, Path
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import importlib.resources
from contextlib import asynccontextmanager
import asyncio
import imaging_server_kit as serverkit
from imaging_server_kit import AlgorithmServer
from imaging_server_kit.core import (
    parse_algo_params_schema,
    encode_contents,
)

templates_dir = importlib.resources.files("imaging_server_kit.core").joinpath(
    "templates"
)
static_dir = importlib.resources.files("imaging_server_kit.core").joinpath("static")

templates = Jinja2Templates(directory=str(templates_dir))

ALGORITHM_HUB_URL = os.getenv("ALGORITHM_HUB_URL", "http://algorithm_hub:8000")
PROCESS_TIMEOUT_SEC = 3600  # Client timeout for the /process route


class MultiAlgorithmServer:  # TODO: could this inherit from algoserver somehow?
    def __init__(self, server_name: str, algorithm_servers: List[AlgorithmServer]):
        self.server_name = server_name
        self.app = FastAPI(title=server_name, lifespan=self.lifespan)

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
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                raise exc
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error_type": "internal_server_error", "message": str(exc)},
            )

        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        self.services = [server_name]

        self.algorithms = {}
        for server in algorithm_servers:
            self.algorithms[server.algorithm_name] = {
                "load_funct": server.load_sample_images,
                "run_funct": server.run_algorithm,
                "parameters_model": server.parameters_model,
                "algo_info": server.algo_info,
                "is_stream": server._is_stream,
            }

        self.register_routes()

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
                "name": self.server_name,
                "url": f"http://{self.server_name}:8000",
            },
        )
        if response.status_code == 201:
            print(f"Service {self.server_name} registered successfully.")
        else:
            print(f"Failed to register {self.server_name}: {response.json()}")

    async def deregister_from_algohub(self):
        deregister_url = f"{ALGORITHM_HUB_URL}/deregister"
        response = requests.post(deregister_url, json={"name": self.server_name})
        if response.status_code == 201:
            print(f"Service {self.server_name} deregistered.")
        else:
            print(f"Failed to deregister {self.server_name}: {response.json()}")

    def register_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            services = list(self.algorithms.keys())
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "services": services,
                },
            )

        @self.app.get("/services")
        def list_services():
            return {"services": list(self.algorithms.keys())}

        @self.app.get("/version")
        def get_version():
            return serverkit.__version__

        @self.app.get("/{algorithm_name}/is_stream")
        def is_stream(algorithm_name):
            if algorithm_name not in self.algorithms:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Algorithm {algorithm_name} not found",
                )
            algo_is_stream = self.algorithms[algorithm_name].get("is_stream")
            return algo_is_stream

        @self.app.get("/{algorithm_name}/info", response_class=HTMLResponse)
        async def get_algorithm_info(
            algorithm_name: str = Path(...),
            request: Request = ...,
        ):
            if algorithm_name not in self.algorithms:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Algorithm {algorithm_name} not found",
                )
            algo_info = self.algorithms[algorithm_name].get("algo_info")
            algo_params_schema = get_algo_params(algorithm_name)
            algo_params = parse_algo_params_schema(algo_params_schema)
            return templates.TemplateResponse(
                "info.html",
                {
                    "request": request,
                    "algo_info": algo_info,
                    "algo_params": algo_params,
                },
            )

        @self.app.post(
            "/{algorithm_name}/process",
            status_code=status.HTTP_201_CREATED,
        )
        async def run_algo(
            algorithm_name: str = Path(...),
            request: Request = ...,
        ):
            if algorithm_name not in self.algorithms:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Algorithm {algorithm_name} not found",
                )

            parameters_model = self.algorithms.get(algorithm_name, {}).get(
                "parameters_model"
            )

            data = await request.json()

            try:
                algo_params = parameters_model(**data)
            except ValidationError as e:
                raise HTTPException(status_code=422, detail=e.errors())

            decoded_params = algo_params.dict()

            return await self._run_algo_logic(algorithm_name, decoded_params)

        @self.app.post(
            "/{algorithm_name}/stream",
            status_code=status.HTTP_201_CREATED,
        )
        async def stream_algo(
            algorithm_name: str = Path(...),
            request: Request = ...,
        ):
            if algorithm_name not in self.algorithms:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Algorithm {algorithm_name} not found",
                )

            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Streaming algorithms in multi-algorithm servers are not supported.",
            )

        @self.app.get("/{algorithm_name}/parameters", response_model=dict)
        def get_algo_params(algorithm_name):
            if algorithm_name not in self.algorithms:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Algorithm {algorithm_name} not found",
                )
            return self.get_algorithm_params(algorithm_name)

        @self.app.get("/{algorithm_name}/sample_images", response_model=dict)
        def get_sample_images(algorithm_name):
            if algorithm_name not in self.algorithms:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Algorithm {algorithm_name} not found",
                )
            return self.get_algorithm_sample_images(algorithm_name)

    async def _serialize_result_tuple(self, result_data_tuple):
        serialized_results = await asyncio.to_thread(
            serverkit.serialize_result_tuple, result_data_tuple
        )
        return serialized_results

    async def _run_algo_logic(self, algorithm_name, algo_params):
        try:
            result_data_tuple = await asyncio.wait_for(
                self._run_algorithm(algorithm_name, **algo_params),
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

        return serialized_results

    async def _run_algorithm(self, algorithm_name: str, **algo_params):
        run_funct = self.algorithms.get(algorithm_name).get("run_funct")
        result_data_tuple = await asyncio.to_thread(run_funct, **algo_params)
        return result_data_tuple

    def get_algorithm_params(self, algorithm_name: str):
        server = self.algorithms.get(algorithm_name)
        parameters_model = server.get("parameters_model")
        return parameters_model.model_json_schema()

    def get_algorithm_sample_images(self, algorithm_name: str):
        server = self.algorithms.get(algorithm_name)
        load_funct = server.get("load_funct")
        images = load_funct()
        encoded_images = [{"sample_image": encode_contents(image)} for image in images]
        return {"sample_images": encoded_images}
