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
from imaging_server_kit.core.errors import InvalidAlgorithmParametersError
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


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@pytest.fixture(scope="module")
def server_url():
    algo_app = AlgorithmApp(algorithms=[add_offset, double_int], name="Test server")
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
    assert set(client.algorithms) == {"add_offset", "double_int"}


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


def test_invalid_params_raise_client_side_error(client):
    image = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(InvalidAlgorithmParametersError):
        client.run(algorithm="add_offset", image=image, offset=999)
