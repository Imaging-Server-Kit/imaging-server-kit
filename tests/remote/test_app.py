"""In-process tests for the AlgorithmApp FastAPI routes.

Uses fastapi.testclient (ASGI transport, no real sockets) so these run fast
and cover the server-side logic (routing, parameter validation, streaming
serialization) in isolation from the real HTTP/msgpack client.
"""

import msgpack
import numpy as np
import pytest
from fastapi.testclient import TestClient

import imaging_server_kit as sk
from imaging_server_kit._version import __version__
from imaging_server_kit.remote.app import AlgorithmApp
from imaging_server_kit.remote.stack_serializer import StackSerializer


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


@pytest.fixture
def client():
    algo_app = AlgorithmApp(algorithms=[add_offset, double_int], name="Test server")
    return TestClient(algo_app.app)


def _decode_process_response(resp) -> sk.Stack:
    """Decode a /process response stream: one msgpack message per layer."""
    unpacker = msgpack.Unpacker(raw=False)
    unpacker.feed(resp.content)
    stack = sk.Stack()
    for serialized_layer in unpacker:
        for layer in StackSerializer().deserialize([serialized_layer], "Python/Napari"):
            stack.add(layer)
    return stack


def _post_process(client, algorithm_name, params_stack):
    payload = StackSerializer().serialize(params_stack, "Python/Napari")
    return client.post(
        f"/{algorithm_name}/process",
        json=payload,
        headers={"User-Agent": "Python/Napari", "accept": "application/msgpack"},
    )


def test_list_algorithms(client):
    resp = client.get("/algorithms")
    assert resp.status_code == 200
    assert set(resp.json()["algorithms"]) == {"add_offset", "double_int"}


def test_version(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.json() == __version__


def test_unknown_algorithm_returns_404(client):
    resp = client.get("/nonexistent/parameters")
    assert resp.status_code == 404


def test_get_parameters(client):
    resp = client.get("/add_offset/parameters")
    assert resp.status_code == 200
    props = resp.json()["properties"]
    assert props["image"]["param_type"] == "image"
    assert props["offset"]["param_type"] == "int"


def test_tileable_flag(client):
    assert client.get("/add_offset/tileable").json() == {"tileable": True}
    assert client.get("/double_int/tileable").json() == {"tileable": False}


def test_signature(client):
    assert client.get("/add_offset/signature").json() == ["image", "offset"]


def test_n_samples_and_sample(client):
    assert client.get("/add_offset/n_samples").json() == {"n_samples": 1}
    assert client.get("/double_int/n_samples").json() == {"n_samples": 0}

    resp = client.get("/add_offset/sample/0")
    assert resp.status_code == 200
    stack = StackSerializer().deserialize(resp.json(), "Python/Napari")
    image_layer = stack.read("image")
    assert image_layer is not None
    assert image_layer.data.shape == (4, 4)


def test_process_scalar_algorithm(client):
    params_stack = sk.Stack()
    params_stack.add(sk.Integer(5, name="x"))
    resp = _post_process(client, "double_int", params_stack)
    assert resp.status_code == 200
    result_stack = _decode_process_response(resp)
    assert result_stack.read("Int").data == 10


def test_process_image_algorithm(client):
    image = np.zeros((4, 4), dtype=np.uint8)
    params_stack = sk.Stack()
    params_stack.add(sk.Image(image, name="image"))
    params_stack.add(sk.Integer(3, name="offset"))
    resp = _post_process(client, "add_offset", params_stack)
    assert resp.status_code == 200
    result_stack = _decode_process_response(resp)
    result_image = result_stack.read("Offset image")
    assert np.array_equal(result_image.data, image + 3)


def test_process_invalid_params_returns_422(client):
    image = np.zeros((2, 2), dtype=np.uint8)
    params_stack = sk.Stack()
    params_stack.add(sk.Image(image, name="image"))
    params_stack.add(sk.Integer(999, name="offset"))  # exceeds max=10
    resp = _post_process(client, "add_offset", params_stack)
    assert resp.status_code == 422
    detail = resp.json()["detail"][0]
    assert detail["loc"][0] == "offset"
