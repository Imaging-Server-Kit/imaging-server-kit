"""
Client interface for the Imaging Server Kit.
"""

import webbrowser
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
import msgpack

from imaging_server_kit.core.runner import AlgorithmRunner, validate_algorithm
from imaging_server_kit.core.errors import (
    AlgorithmRuntimeError,
    AlgorithmServerError,
    AlgorithmTimeoutError,
    InvalidAlgorithmParametersError,
)
from imaging_server_kit.core.stack import Stack
from imaging_server_kit.remote.stack_serializer import StackSerializer
from imaging_server_kit.remote.serializer import ERROR_FRAME_KEY

# Unlimited input size - Implies trusted input. TODO: should this be made more clear (or configurable)?
MAX_BUFFER_SIZE = 0

# Timeout (in seconds) for the requests to the metadata endpoints (not for running algorithms),
# so that an unresponsive server raises an error instead of hanging forever.
GET_TIMEOUT = 60


class ServerRequestError(Exception):
    """Exception raised when HTTP requests fail."""

    def __init__(self, url: str, error: Exception, message="Request to server failed"):
        self.url = url
        self.error = error
        self.message = f"{message} ({url=}): {error}"
        super().__init__(self.message)


class Client(AlgorithmRunner):
    """Client to connect to algorithm servers.

    A client implements the shared
    [`AlgorithmRunner`][imaging_server_kit.AlgorithmRunner] interface, so the same
    code can run an algorithm locally or on a server. Computations requested with
    `run()` are executed on the server.

    Parameters
    ----------
    server_url : str, optional
        Address of the algorithm server, e.g. `"http://localhost:8000"`. If provided,
        the client connects to the server immediately.
    name : str, default="client"
        A name identifying the client.

    Attributes
    ----------
    server_url : str
        Address of the algorithm server.
    algorithms : list of str
        The names of the algorithms available on the server.

    Examples
    --------
    >>> client = sk.Client("http://localhost:8000")
    >>> results = client.run(image, threshold=100)
    """

    def __init__(self, server_url: Optional[str] = None, name: str = "client") -> None:
        self.server_url = server_url
        self._algorithms = []
        if server_url:
            self.connect(server_url)
        self._name = name

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
    def server_url(self) -> Optional[str]:
        return self._server_url

    @server_url.setter
    def server_url(self, server_url: Optional[str]):
        self._server_url = server_url

    def connect(self, server_url: str) -> None:
        """Connect to an algorithm server and list its algorithms.

        Parameters
        ----------
        server_url : str
            Address of the algorithm server, e.g. `"http://localhost:8000"`.
        """
        self.server_url = server_url.rstrip("/")
        endpoint = urljoin(self.server_url + "/", "algorithms")
        json_response = self._access_algo_get_endpoint(endpoint)
        self.algorithms = json_response.get("algorithms")

    @validate_algorithm
    def info(self, algorithm: Optional[str] = None):
        webbrowser.open(f"{self.server_url}/{algorithm}/info")

    @validate_algorithm
    def get_parameters(self, algorithm: Optional[str] = None) -> Dict:
        endpoint = f"{self.server_url}/{algorithm}/parameters"
        return self._access_algo_get_endpoint(endpoint)

    @validate_algorithm
    def get_sample(self, algorithm: Optional[str] = None, idx: int = 0) -> Stack:
        n_samples = self.get_n_samples(algorithm)
        if (idx < 0) or (idx > n_samples - 1):
            raise ValueError(
                f"Algorithm provides {n_samples} samples. Max value for `idx` is {n_samples-1}!"
            )
        endpoint = f"{self.server_url}/{algorithm}/sample/{idx}"
        serialized_sample_stack = self._access_algo_get_endpoint(endpoint)
        sample_stack = StackSerializer.deserialize(serialized_sample_stack)
        return sample_stack

    @validate_algorithm
    def get_n_samples(self, algorithm: Optional[str] = None) -> int:
        endpoint = f"{self.server_url}/{algorithm}/n_samples"
        json_response = self._access_algo_get_endpoint(endpoint)
        n_samples = json_response.get("n_samples")
        return n_samples

    @validate_algorithm
    def is_tileable(self, algorithm: Optional[str] = None) -> bool:
        endpoint = f"{self.server_url}/{algorithm}/tileable"
        json_response = self._access_algo_get_endpoint(endpoint)
        is_tileable = json_response.get("tileable")
        return is_tileable

    @validate_algorithm
    def get_signature_params(self, algorithm: Optional[str] = None) -> List[str]:
        endpoint = f"{self.server_url}/{algorithm}/signature"
        return self._access_algo_get_endpoint(endpoint)

    def _stream(self, algorithm, params_stack: Stack):
        endpoint = f"{self.server_url}/{algorithm}/process"
        with requests.Session() as client:
            try:
                response = client.post(
                    endpoint,
                    json=StackSerializer.serialize(params_stack),
                    headers={
                        "Content-Type": "application/json",
                        "accept": "application/msgpack",
                    },
                    stream=True,
                    # TODO: We *could* implement a timeout here, but not sure what's the best strategy for that, so we leave it as todo.
                    # timeout=3600,
                )
            except requests.RequestException as e:
                raise ServerRequestError(endpoint, e)

            if response.status_code == 200:
                unpacker = msgpack.Unpacker(raw=False, max_buffer_size=MAX_BUFFER_SIZE)

                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue

                    unpacker.feed(chunk)

                    for serialized_stack in unpacker:
                        if ERROR_FRAME_KEY in serialized_stack:
                            error = serialized_stack[ERROR_FRAME_KEY]
                            raise AlgorithmRuntimeError(
                                algorithm=algorithm,
                                error=RuntimeError(f"{error['type']}: {error['message']}"),
                                message="Algorithm did not run successfully on the server. ",
                            )
                        yield StackSerializer.deserialize([serialized_stack])
            else:
                self._handle_response_errored(response)

    def _access_algo_get_endpoint(self, endpoint: str):
        with requests.Session() as client:
            try:
                response = client.get(endpoint, timeout=GET_TIMEOUT)
            except requests.RequestException as e:
                raise ServerRequestError(endpoint, e)
        if response.status_code == 200:
            return response.json()
        else:
            self._handle_response_errored(response)

    def _handle_response_errored(self, response):
        if response.status_code == 422:
            try:
                response_body = response.json()
            except ValueError:
                response_body = response.text
            raise InvalidAlgorithmParametersError(response.status_code, response_body)
        elif response.status_code == 504:
            raise AlgorithmTimeoutError(response.status_code, response.text)
        else:
            raise AlgorithmServerError(response.status_code, response.text)
