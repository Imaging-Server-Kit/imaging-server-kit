"""
Client interface for the Imaging Server Kit.
"""

import webbrowser
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests
import msgpack

from imaging_server_kit.core.runner import AlgorithmRunner, validate_algorithm
from imaging_server_kit.core.errors import (
    AlgorithmServerError,
    AlgorithmTimeoutError,
    InvalidAlgorithmParametersError,
    ServerRequestError,
)
from imaging_server_kit.core.results import Results


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
        sample_results = Results.deserialize(
            serialized_sample_results, client_origin="Python/Napari"
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

    def _stream(self, algorithm, params_res: Results):
        endpoint = f"{self.server_url}/{algorithm}/process"
        with requests.Session() as client:
            try:
                response = client.post(
                    endpoint,
                    json=params_res.serialize("Python/Napari"),
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
                    for serialized_results in unpacker:
                        yield Results.deserialize([serialized_results], "Python/Napari")
            else:
                self._handle_response_errored(response)

    def _access_algo_get_endpoint(self, endpoint: str):
        with requests.Session() as client:
            try:
                response = client.get(endpoint)
            except requests.RequestException as e:
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
