from typing import Dict, List, Tuple

import msgpack
import numpy as np
import requests
from urllib.parse import urljoin
from tqdm import tqdm

import httpx

from imaging_server_kit.core import (
    AlgorithmNotFoundError,
    AlgorithmServerError,
    AlgorithmTimeoutError,
    InvalidAlgorithmParametersError,
    ServerRequestError,
    decode_contents,
    deserialize_result_tuple,
    encode_contents,
)
from imaging_server_kit.streaming import initialize_tiled_image


def validate_algorithm(func):
    def wrapper(self, algorithm=None, *args, **kwargs):
        # algorithm = self._validate_algorithm(algorithm)
        if algorithm is None:
            if len(self.algorithms) > 0:
                algorithm = self.algorithms[0]
            else:
                raise AlgorithmNotFoundError(algorithm)
        else:
            if algorithm not in self.algorithms:
                raise AlgorithmNotFoundError(algorithm)

        return func(self, algorithm, *args, **kwargs)
    return wrapper


class AlgorithmStreamError(Exception):
    """Exception raised when an algorithm is incorrectly used as a stream."""

    def __init__(
        self, algorithm_name, message="Algorithm incorrectly used as a stream."
    ):
        self.algorithm_name = algorithm_name
        self.message = f"{algorithm_name}: {message}"
        super().__init__(self.message)


class Client:
    def __init__(self, server_url: str = "") -> None:
        self.server_url = server_url
        self._algorithms = {}
        if server_url:
            self.connect(server_url)
        self.token = None
        self.client = None

    def connect(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.client = httpx.Client(base_url=self.server_url)

        endpoint = urljoin(self.server_url + "/", "services")
        try:
            response = self.client.get(endpoint)
        except httpx.RequestError as e:
            raise ServerRequestError(endpoint, e)

        if response.status_code == 200:
            self.algorithms = response.json().get("services")
        else:
            raise AlgorithmServerError(response.status_code, response.text)

    def login(self, username, password):
        endpoint = f"{self.server_url}/auth/jwt/login"
        try:
            response = self.client.post(
                endpoint,
                data={
                    "username": username,
                    "password": password,
                },
            )
        except httpx.RequestError as e:
            raise ServerRequestError(endpoint, e)

        if response.status_code == 200:
            token = response.json().get("access_token")
            self.token = token
        else:
            raise AlgorithmServerError(response.status_code, response.text)

    @property
    def server_url(self) -> str:
        return self._server_url

    @server_url.setter
    def server_url(self, server_url: str):
        self._server_url = server_url

    @property
    def algorithms(self) -> Dict[str, str]:
        return self._algorithms

    @algorithms.setter
    def algorithms(self, algorithms: Dict[str, str]):
        self._algorithms = algorithms

    @validate_algorithm
    def run_algorithm(self, algorithm=None, **algo_params) -> List[Tuple]:
        # algorithm = self._validate_algorithm(algorithm)

        if self.is_algo_stream(algorithm):
            raise AlgorithmStreamError(
                algorithm,
                message="Algorithm is a stream. Use client.stream_algorithm() or client.consume_stream() instead.",
            )

        algo_params_encoded = self._encode_numpy_parameters(algo_params)
        try:
            endpoint = f"{self.server_url}/{algorithm}/process"
            response = self.client.post(
                endpoint,
                json=algo_params_encoded,
                headers={
                    "Content-Type": "application/json",
                    "accept": "application/json",
                    "Authorization": f"Bearer {self.token}",
                },
                stream=False,
            )
        except httpx.RequestError as e:
            raise ServerRequestError(endpoint, e)

        if response.status_code == 201:
            serialized_results = response.json()
            # serialized_results = [msgpack.unpackb(r) for r in serialized_results]
            result_data_tuple = deserialize_result_tuple(serialized_results)
            return result_data_tuple
        else:
            self._handle_response_errored(response)

    @validate_algorithm
    def stream_algorithm(self, algorithm=None, **algo_params):
        if not self.is_algo_stream(algorithm):
            raise AlgorithmStreamError(
                algorithm,
                message="Algorithm is not a stream. Use client.run_algorithm() instead.",
            )

        algo_params_encoded = self._encode_numpy_parameters(algo_params)

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
                    # timeout=None,  # for httpx
                    stream=True,
                )
            except httpx.RequestError as e:
                raise ServerRequestError(endpoint, e)
            
            if response.status_code == 200:
                unpacker = msgpack.Unpacker(raw=False)

                # for chunk in response.iter_bytes():  # For httpx
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    unpacker.feed(chunk)
                    result_data_tuple = deserialize_result_tuple(unpacker)  # TODO: Introduce a try/except around deserialization to avoid hidden silent failures.
                    for data, data_params, data_type in result_data_tuple:
                        yield [(data, data_params, data_type)]
            else:
                self._handle_response_errored(response)

    @validate_algorithm
    def consume_stream(self, algorithm=None, **algo_params) -> List[Tuple]:
        """(Experimental) Consumes an algorithm streamed in tiles and shows a progress bar."""
        if not self.is_algo_stream(algorithm):
            raise AlgorithmStreamError(
                algorithm,
                message="Algorithm is not a stream. Use client.run_algorithm() instead.",
            )

        first_tile = True
        progress = tqdm(total=None)
        for result_data_tuple in self.stream_algorithm(algorithm, **algo_params):
            data, meta_data, data_type = result_data_tuple[0]

            tile_params = meta_data.get("tile_params")
            if not tile_params:
                print(f"{algorithm} is not a tiled streamed algorithm.")
                return

            if data_type not in [
                "image",
                "mask",
                "instance_mask",
                "mask3d",
            ]:
                print(f"{data_type} in streaming mode isn't supported yet.")
                return

            # The only `interesting` case - when we get a progress bar and reconstructed result at the end
            if first_tile:
                image_layer_data = initialize_tiled_image(tile_params)
                first_tile = False

            if data_type == "instance_mask":
                # TODO: implement a good enough merging strategy for instance masks (difficult!)
                mask = data == 0
                data += image_layer_data.max()
                data[mask] = 0

            chunk_pos_x = tile_params.get("pos_x")
            chunk_pos_y = tile_params.get("pos_y")

            try:
                if tile_params.get("pos_z"):  # 3D case
                    chunk_pos_z = tile_params.get("pos_z")
                    chunk_size_z, chunk_size_y, chunk_size_x = (
                        data.shape[0],
                        data.shape[1],
                        data.shape[2],
                    )
                    image_layer_data[
                        chunk_pos_z : (chunk_pos_z + chunk_size_z),
                        chunk_pos_y : (chunk_pos_y + chunk_size_y),
                        chunk_pos_x : (chunk_pos_x + chunk_size_x),
                    ] = data
                else:  # 2D / RGB cases
                    chunk_size_x, chunk_size_y = (
                        data.shape[0],
                        data.shape[1],
                    )
                    image_layer_data[
                        chunk_pos_x : (chunk_pos_x + chunk_size_x),
                        chunk_pos_y : (chunk_pos_y + chunk_size_y),
                    ] = data
            except:
                print("Attempted to write tiles outside of the image.")

        if progress.total is None:
            tile_params = meta_data.get("tile_params")
            if tile_params:
                n_tiles = tile_params.get("n_tiles")
                progress.total = n_tiles
                progress.refresh()
        progress.update(1)
        
        return (image_layer_data, meta_data, data_type)

    @validate_algorithm
    def is_algo_stream(self, algorithm=None) -> bool:
        # algorithm = self._validate_algorithm(algorithm)

        endpoint = f"{self.server_url}/{algorithm}/is_stream"

        try:
            response = self.client.get(endpoint)
        except httpx.RequestError as e:
            raise ServerRequestError(endpoint, e)

        if response.status_code == 200:
            return response.json()
        else:
            self._handle_response_errored(response)

    @validate_algorithm
    def get_algorithm_parameters(self, algorithm=None) -> Dict:
        # algorithm = self._validate_algorithm(algorithm)

        endpoint = f"{self.server_url}/{algorithm}/parameters"

        try:
            response = self.client.get(endpoint)
        except httpx.RequestError as e:
            raise ServerRequestError(endpoint, e)

        if response.status_code == 200:
            return response.json()
        else:
            self._handle_response_errored(response)

    @validate_algorithm
    def get_sample_images(
        self, algorithm=None, first_only: bool = False
    ) -> List["np.ndarray"]:
        # algorithm = self._validate_algorithm(algorithm)

        endpoint = f"{self.server_url}/{algorithm}/sample_images"

        try:
            response = self.client.get(endpoint)
        except httpx.RequestError as e:
            raise ServerRequestError(endpoint, e)

        if response.status_code == 200:
            images = []
            for content in response.json().get("sample_images"):
                encoded_image = content.get("sample_image")
                image = decode_contents(encoded_image)
                images.append(image)
                if first_only:
                    return image
            return images
        else:
            self._handle_response_errored(response)

    # def _validate_algorithm(
    #     self, algorithm=None
    # ) -> str:  # TODO: Could this be made into a decorator @validate_algorithm instead?
    #     if algorithm is None:
    #         if len(self.algorithms) > 0:
    #             algorithm = self.algorithms[0]
    #         else:
    #             raise AlgorithmNotFoundError(algorithm)
    #     else:
    #         if algorithm not in self.algorithms:
    #             raise AlgorithmNotFoundError(algorithm)
    #     return algorithm

    def _encode_numpy_parameters(self, algo_params: dict) -> dict:
        return {
            param: encode_contents(value) if isinstance(value, np.ndarray) else value
            for param, value in algo_params.items()
        }

    def _handle_response_errored(self, response):
        if response.status_code == 422:
            raise InvalidAlgorithmParametersError(response.status_code, response.json())
        elif response.status_code == 504:
            raise AlgorithmTimeoutError(response.status_code, response.text)
        else:
            raise AlgorithmServerError(response.status_code, response.text)
