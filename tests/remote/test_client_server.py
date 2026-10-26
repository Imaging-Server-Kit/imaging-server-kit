"""End-to-end tests: a real uvicorn server driven by the real Client.

Unlike test_app.py (in-process, ASGI transport), these tests exercise the
actual code Client._stream() uses to talk to a server: real sockets, real
HTTP, real msgpack streaming, and the server/client error-mapping contract.
Kept to a small number of high-value cases; route-level edge cases belong
in test_app.py.
"""

import socket
import threading
import time

import numpy as np
import pytest
import uvicorn

import imaging_server_kit as sk
from imaging_server_kit.core.errors import (
    AlgorithmNotFoundError,
    AlgorithmRuntimeError,
    AlgorithmServerError,
    AlgorithmTimeoutError,
    InvalidAlgorithmParametersError,
    ServerRequestError,
)
from imaging_server_kit.remote.app import AlgorithmApp
from imaging_server_kit.remote.client import Client


@sk.algorithm(
    name="add_offset",
    parameters={
        "image": sk.Image(),
        "offset": sk.Integer(default=1, min=0, max=10),
    },
    samples=[{"image": np.arange(16, dtype=np.uint8).reshape(4, 4), "offset": 2}],
    tileable=True,
)
def add_offset(image, offset):
    return sk.Image(image + offset, name="Offset image")


@sk.algorithm(name="double_int", parameters={"x": sk.Integer(default=1)})
def double_int(x):
    return sk.Integer(x * 2)


@sk.algorithm(name="fails", parameters={"x": sk.Integer(default=1)})
def fails(x):
    raise RuntimeError("boom")


@sk.algorithm(name="fails_after_yield", parameters={"x": sk.Integer(default=1)})
def fails_after_yield(x):
    yield sk.Integer(x)
    raise RuntimeError("boom")


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(scope="module")
def server_url():
    algo_app = AlgorithmApp(
        algorithms=[add_offset, double_int, fails, fails_after_yield], name="Test server"
    )
    port = _free_port()

    config = uvicorn.Config(algo_app.app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn server failed to start in time"

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def client(server_url):
    return Client(server_url)


def test_connect_lists_algorithms(client):
    assert set(client.algorithms) == {"add_offset", "double_int", "fails", "fails_after_yield"}


def test_run_scalar_algorithm_matches_local(client):
    remote_out = client.run(algorithm="double_int", x=5)
    local_out = double_int.run(x=5)
    assert remote_out[0].data == local_out[0].data == 10


def test_run_image_algorithm_matches_local(client):
    image = (np.arange(64) % 50).astype(np.uint8).reshape(8, 8)

    remote_out = client.run(algorithm="add_offset", image=image, offset=3)
    local_out = add_offset.run(image=image, offset=3)

    assert np.array_equal(remote_out.read("Offset image").data, image + 3)
    assert np.array_equal(
        remote_out.read("Offset image").data, local_out.read("Offset image").data
    )


def test_run_tiled_matches_local_tiled(client):
    image = (np.arange(400) % 50).astype(np.uint8).reshape(20, 20)

    remote_out = client.run(
        algorithm="add_offset", image=image, offset=5, tiled=True, tile_size=8
    )
    local_out = add_offset.run(image=image, offset=5, tiled=True, tile_size=8)

    assert np.array_equal(remote_out.read("Offset image").data, image + 5)
    assert np.array_equal(
        remote_out.read("Offset image").data, local_out.read("Offset image").data
    )


def test_get_sample_roundtrip(client):
    sample_stack = client.get_sample(algorithm="add_offset", idx=0)
    image_layer = sample_stack.read("image")
    assert image_layer is not None
    assert image_layer.data.shape == (4, 4)


def test_get_sample_out_of_range_or_missing(client):
    with pytest.raises(ValueError):
        client.get_sample(algorithm="add_offset", idx=-1)
    assert client.get_sample(algorithm="double_int") is None


def test_invalid_params_raise_client_side_error(client):
    image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(InvalidAlgorithmParametersError):
        client.run(algorithm="add_offset", image=image, offset=999)


@pytest.mark.parametrize("algorithm", ["fails", "fails_after_yield"])
def test_server_side_error_is_raised_by_client(client, algorithm):
    """The original error message reaches the client, and the server keeps working."""
    with pytest.raises(AlgorithmRuntimeError, match="RuntimeError: boom"):
        client.run(algorithm=algorithm)

    assert client.run(algorithm="double_int", x=2)[0].data == 4


def test_unknown_algorithm_raises(client):
    with pytest.raises(AlgorithmNotFoundError):
        client.run(algorithm="does_not_exist")


### Error mapping (no server needed) ###


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.text = str(body)
        self._body = body

    def json(self):
        if not isinstance(self._body, dict):
            raise ValueError("Not JSON")
        return self._body


@pytest.mark.parametrize(
    "status_code, error_cls",
    [
        (422, InvalidAlgorithmParametersError),
        (504, AlgorithmTimeoutError),
        (500, AlgorithmServerError),
        (404, AlgorithmServerError),
    ],
)
def test_response_status_codes_map_to_errors(status_code, error_cls):
    with pytest.raises(error_cls) as exc_info:
        Client()._handle_response_errored(_FakeResponse(status_code, "Server says no"))

    assert exc_info.value.status_code == status_code


def test_invalid_parameters_error_reports_the_failing_parameter():
    body = {"detail": [{"loc": ["offset"], "msg": "Input should be less than 10", "input": 12}]}

    with pytest.raises(InvalidAlgorithmParametersError, match="offset.*less than 10.*12"):
        Client()._handle_response_errored(_FakeResponse(422, body))


def test_unreachable_server_raises_server_request_error():
    with pytest.raises(ServerRequestError):
        Client(f"http://127.0.0.1:{_free_port()}")
