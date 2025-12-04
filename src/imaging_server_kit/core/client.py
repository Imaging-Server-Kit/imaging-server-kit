"""
Client interface for the Imaging Server Kit.
"""

import webbrowser
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
import httpx
import msgpack

from imaging_server_kit.core.runner import AlgorithmRunner, validate_algorithm
from imaging_server_kit.core.errors import (
    AlgorithmServerError,
    AlgorithmTimeoutError,
    InvalidAlgorithmParametersError,
    ServerRequestError,
)
import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.results import Results
from imaging_server_kit.core.serialization import deserialize_results


TIMEOUT_SEC = 3600  # Timeout for the /process route (in seconds)


class Client(AlgorithmRunner):
    """Client to connect to and interact with algorithm servers.
    
    Attributes
    ----------
    server_url: Address of the algorithm server.
    algorithms: A list of available algorithms.

    Methods
    ----------
    connect(): Connect to an algorithm server.
    run(): Execute the algorithm with a set of parameters.
        Set `tiled=True` for tiled inference. 
        Raises a ValidationError when parameters are invalidated.
    get_n_samples(): Get the number of samples available.
    get_sample(): Get a sample by index.
    info(): Access algorithm documentation.
    get_parameters(): Get the algorithm parameters schema.    
    """
    def __init__(self, server_url: Optional[str] = None) -> None:
        self.server_url = server_url
        self._algorithms = []
        if server_url:
            self.connect(server_url)
        self.token = None

    @property
    def algorithms(self) -> Iterable[str]:
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms: Iterable[str]):
        self._algorithms = algorithms

    @property
    def server_url(self) -> Optional[str]:
        return self._server_url

    @server_url.setter
    def server_url(self, server_url: Optional[str]):
        self._server_url = server_url

    def connect(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")
        endpoint = urljoin(self.server_url + "/", "algorithms")
        json_response = self._access_algo_get_endpoint(endpoint)
        self.algorithms = json_response.get("algorithms")

    @validate_algorithm
    def info(self, algorithm=None):
        webbrowser.open(f"{self.server_url}/{algorithm}/info")

    @validate_algorithm
    def get_parameters(self, algorithm=None) -> Dict:
        endpoint = f"{self.server_url}/{algorithm}/parameters"
        return self._access_algo_get_endpoint(endpoint)

    @validate_algorithm
    def get_sample(self, algorithm=None, idx: int = 0) -> Results:
        n_samples = self.get_n_samples(algorithm)
        if (idx < 0) | (idx > n_samples - 1):
            raise ValueError(
                f"Algorithm provides {n_samples} samples. Max value for `idx` is {n_samples-1}!"
            )
        endpoint = f"{self.server_url}/{algorithm}/sample/{idx}"
        serialized_sample_results = self._access_algo_get_endpoint(endpoint)
        sample_results = deserialize_results(
            serialized_sample_results,
            client_origin="Python/Napari",
        )
        return sample_results

    @validate_algorithm
    def get_n_samples(self, algorithm=None) -> int:
        endpoint = f"{self.server_url}/{algorithm}/n_samples"
        json_response = self._access_algo_get_endpoint(endpoint)
        n_samples = json_response.get("n_samples")
        return n_samples
    
    @validate_algorithm
    def is_tileable(self, algorithm=None) -> bool:
        endpoint = f"{self.server_url}/{algorithm}/tileable"
        json_response = self._access_algo_get_endpoint(endpoint)
        is_tileable = json_response.get("tileable")
        return is_tileable

    @validate_algorithm
    def get_signature_params(self, algorithm: str) -> List[str]:
        endpoint = f"{self.server_url}/{algorithm}/signature"
        return self._access_algo_get_endpoint(endpoint)

    def _is_stream(self, algorithm=None) -> bool:
        endpoint = f"{self.server_url}/{algorithm}/is_stream"
        return self._access_algo_get_endpoint(endpoint)

    def _stream(self, algorithm, param_results: Results):
        endpoint = f"{self.server_url}/{algorithm}/stream"
        with requests.Session() as client:
            try:
                response = client.post(
                    endpoint,
                    json=param_results.serialize("Python/Napari"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.token}",
                        "accept": "application/msgpack",
                        "User-Agent": "Python/Napari",
                    },
                    stream=True,
                )
            except requests.RequestException as e:
                raise ServerRequestError(endpoint, e)

            if response.status_code == 200:
                unpacker = msgpack.Unpacker(raw=False)
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    unpacker.feed(chunk)
                    yield deserialize_results(unpacker, "Python/Napari")
            else:
                self._handle_response_errored(response)

    def _tile(
        self,
        algorithm: str,
        tile_size_px: int,
        overlap_percent: float,
        delay_sec: float,
        randomize: bool,
        param_results: Results,
    ):
        """Breaks down the 2D image parameter into tiles before sequentially postiong to /process."""
        endpoint = f"{self.server_url}/{algorithm}/process"
        with requests.Session() as client:
            for algo_params_tile, tile_info in etc.generate_tiles(
                param_results,
                tile_size_px,
                overlap_percent,
                delay_sec,
                randomize,
            ):
                try:
                    response = client.post(
                        endpoint,
                        json=algo_params_tile.serialize("Python/Napari"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.token}",
                            "accept": "application/msgpack",
                            "User-Agent": "Python/Napari",
                        },
                    )
                except requests.RequestException as e:
                    raise ServerRequestError(endpoint, e)

                if response.status_code == 201:
                    results = deserialize_results(response.json(), "Python/Napari")
                    for layer in results:
                        layer.meta = layer.meta | tile_info
                    yield results
                else:
                    self._handle_response_errored(response)

    def _run(self, algorithm: str, param_results: Results) -> Results:
        endpoint = f"{self.server_url}/{algorithm}/process"
        with httpx.Client(base_url=self.server_url, timeout=TIMEOUT_SEC) as client:
            try:
                response = client.post(
                    endpoint,
                    json=param_results.serialize("Python/Napari"),
                    headers={
                        "Content-Type": "application/json",
                        "accept": "application/json",
                        "Authorization": f"Bearer {self.token}",
                        "User-Agent": "Python/Napari",
                    },
                )
            except httpx.RequestError as e:
                raise ServerRequestError(endpoint, e)
        if response.status_code == 201:
            return deserialize_results(response.json(), "Python/Napari")
        else:
            self._handle_response_errored(response)

    def _access_algo_get_endpoint(self, endpoint: str):
        """Used to get /parameters, /is_stream"""
        with httpx.Client(base_url=self.server_url) as client:
            try:
                response = client.get(endpoint)
            except httpx.RequestError as e:
                raise ServerRequestError(endpoint, e)
        if response.status_code == 200:
            return response.json()
        else:
            self._handle_response_errored(response)

    def _handle_response_errored(self, response):
        if response.status_code == 422:
            raise InvalidAlgorithmParametersError(response.status_code, response.json())
        elif response.status_code == 504:
            raise AlgorithmTimeoutError(response.status_code, response.text)
        else:
            raise AlgorithmServerError(response.status_code, response.text)
