"""
Client interface for the Imaging Server Kit.
"""

import webbrowser
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urljoin
import numpy as np

import requests
import httpx
import msgpack

from imaging_server_kit.core.runner import AlgorithmRunner, validate_algorithm
from imaging_server_kit.core.encoding import decode_contents, encode_contents
from imaging_server_kit.core.errors import (
    AlgorithmServerError,
    AlgorithmTimeoutError,
    InvalidAlgorithmParametersError,
    ServerRequestError,
)
import imaging_server_kit.core._etc as etc
from imaging_server_kit.core.serialization import deserialize_results


def _encode_numpy_parameters(algo_params: Dict) -> Dict:
    return {
        param: encode_contents(value) if isinstance(value, np.ndarray) else value
        for param, value in algo_params.items()
    }


def _decode_numpy_parameters(algo_params: Dict, algo_param_defs: Dict) -> Dict:
    param_keys = list(algo_params.keys())
    param_values = list(algo_params.values())

    param_types = []
    for key in param_keys:
        algo_def = algo_param_defs.get(key)
        if algo_def:
            param_types.append(algo_def.get("param_type"))

    numpy_types = [
        "image",
        "mask",
        "instance_mask",
        "points",
        "vectors",
        "boxes",
        "paths",
        "tracks",
    ]
    
    decoded_params = {}
    for param_key, param_val, param_type in zip(param_keys, param_values, param_types):
        if param_type in numpy_types:
            val = decode_contents(param_val)
        else:
            val = param_val
        decoded_params[param_key] = val

    return decoded_params


class Client(AlgorithmRunner):
    def __init__(self, server_url: str = "") -> None:
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
    def server_url(self) -> str:
        return self._server_url

    @server_url.setter
    def server_url(self, server_url: str):
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
    def get_sample(self, algorithm=None, idx: int = 0) -> Dict[str, Any]:
        n_samples = self.get_n_samples(algorithm)
        if (idx < 0) | (idx > n_samples-1):
            raise ValueError(f"Algorithm provides {n_samples} samples. Max value for `idx` is {n_samples-1}!")
        
        endpoint = f"{self.server_url}/{algorithm}/sample/{idx}"
        json_response = self._access_algo_get_endpoint(endpoint)
        algo_param_defs = self.get_parameters(algorithm).get("properties")
        decoded_params = _decode_numpy_parameters(json_response, algo_param_defs)
        return decoded_params
    
    @validate_algorithm
    def get_n_samples(self, algorithm=None) -> int:
        endpoint = f"{self.server_url}/{algorithm}/n_samples"
        json_response = self._access_algo_get_endpoint(endpoint)
        n_samples = json_response.get("n_samples")
        return n_samples

    @validate_algorithm
    def get_signature_params(self, algorithm: str) -> List[str]:
        endpoint = f"{self.server_url}/{algorithm}/signature"
        return self._access_algo_get_endpoint(endpoint)

    def _is_stream(self, algorithm=None) -> bool:
        endpoint = f"{self.server_url}/{algorithm}/is_stream"
        return self._access_algo_get_endpoint(endpoint)

    def _stream(self, algorithm, **algo_params):
        algo_params_encoded = _encode_numpy_parameters(algo_params)
        endpoint = f"{self.server_url}/{algorithm}/stream"
        with requests.Session() as client:
            try:
                response = client.post(
                    endpoint,
                    json=algo_params_encoded,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.token}",
                        "accept": "application/msgpack",
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
                    yield deserialize_results(unpacker)
            else:
                self._handle_response_errored(response)

    def _tile(
        self,
        algorithm,
        tile_size_px,
        overlap_percent,
        delay_sec,
        randomize,
        **algo_params,
    ):
        """Breaks down the 2D image parameter into tiles before sequentially postiong to /process."""
        with requests.Session() as client:
            algo_param_defs = self.get_parameters(algorithm).get("properties")

            for algo_params_tile, tile_info in etc.generate_tiles(
                algo_param_defs,
                algo_params,
                tile_size_px,
                overlap_percent,
                delay_sec,
                randomize,
            ):
                algo_params_tile_encoded = _encode_numpy_parameters(algo_params_tile)
                endpoint = f"{self.server_url}/{algorithm}/process"
                try:
                    response = client.post(
                        endpoint,
                        json=algo_params_tile_encoded,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.token}",
                            "accept": "application/msgpack",
                        },
                    )
                except requests.RequestException as e:
                    raise ServerRequestError(endpoint, e)

                if response.status_code == 201:
                    results = deserialize_results(response.json())
                    for layer in results:
                        layer.meta = layer.meta | tile_info
                    yield results
                else:
                    self._handle_response_errored(response)
            return []

    def _run(self, algorithm, **algo_params) -> Iterable[Tuple]:
        algo_params_encoded = _encode_numpy_parameters(algo_params)
        endpoint = f"{self.server_url}/{algorithm}/process"
        with httpx.Client(base_url=self.server_url) as client:
            try:
                response = client.post(
                    endpoint,
                    json=algo_params_encoded,
                    headers={
                        "Content-Type": "application/json",
                        "accept": "application/json",
                        "Authorization": f"Bearer {self.token}",
                    },
                )
            except httpx.RequestError as e:
                raise ServerRequestError(endpoint, e)
        if response.status_code == 201:
            return deserialize_results(response.json())
        else:
            self._handle_response_errored(response)

    def _access_algo_get_endpoint(self, endpoint):
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
