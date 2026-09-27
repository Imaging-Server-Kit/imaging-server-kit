"""Round trips through the wire format: serialize => transport (JSON or msgpack) => deserialize.

Requests (client => server) are sent as JSON, responses (server => client) as msgpack.
Every layer kind must survive both, with its data (values and dtype), meta, position and tile meta.
"""

import json

import msgpack
import numpy as np
import pytest

import imaging_server_kit as sk
from imaging_server_kit.core.tiling import TileMeta
from imaging_server_kit.remote.layer_serializer import LayerSerializer
from imaging_server_kit.remote.stack_serializer import StackSerializer

TRANSPORTS = {
    "json": lambda obj: json.loads(json.dumps(obj)),
    "msgpack": lambda obj: msgpack.unpackb(msgpack.packb(obj), raw=False),
}


@pytest.fixture(params=list(TRANSPORTS))
def transport(request):
    return TRANSPORTS[request.param]


def _roundtrip(layer: sk.Layer, transport) -> sk.Layer:
    return LayerSerializer.deserialize(transport(LayerSerializer.serialize(layer)))


def _normalize(obj):
    """Normalize values that the wire format legitimately changes (tuples => lists, Numpy scalars => Python)."""
    if isinstance(obj, np.ndarray):
        return ("ndarray", str(obj.dtype), obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def _assert_same_layer(result: sk.Layer, layer: sk.Layer) -> None:
    assert type(result) is type(layer)
    assert result.name == layer.name
    assert _normalize(result.data) == _normalize(layer.data)  # Values and dtypes
    assert _normalize(result.meta) == _normalize(layer.meta)
    assert result.tile_meta.serialize() == layer.tile_meta.serialize()


def _rng():
    return np.random.default_rng(0)


# Values are exactly representable in float32 (objects are sent as float32)
LAYERS = {
    "image-uint8": lambda: sk.Image(_rng().integers(0, 255, (6, 5)).astype(np.uint8)),
    "image-uint16": lambda: sk.Image(_rng().integers(0, 60000, (6, 5)).astype(np.uint16)),
    "image-float32": lambda: sk.Image(_rng().random((6, 5)).astype(np.float32)),
    "image-float64": lambda: sk.Image(_rng().random((6, 5))),
    "image-rgb": lambda: sk.Image(_rng().integers(0, 255, (6, 5, 3)).astype(np.uint8), rgb=True),
    "image-channel_axis=0": lambda: sk.Image(_rng().random((2, 6, 5)), channel_axis=0),
    "mask-int32": lambda: sk.Mask(_rng().integers(0, 100_000, (6, 5)).astype(np.int32)),
    "mask-uint32": lambda: sk.Mask(_rng().integers(0, 100_000, (6, 5)).astype(np.uint32)),
    "points": lambda: sk.Points(np.array([[1.0, 2.5], [30.0, 4.0]])),
    "points-3d": lambda: sk.Points(np.array([[1.0, 2.5, 3.0]])),
    "boxes": lambda: sk.Boxes(np.array([[[0.0, 0.0], [0.0, 4.0], [3.0, 4.0], [3.0, 0.0]]])),
    "vectors": lambda: sk.Vectors(np.array([[[1.0, 2.0], [0.5, -1.5]]])),
    "paths": lambda: sk.Paths([np.array([[0.0, 0.0], [1.0, 2.0]]), np.array([[5.0, 5.0], [6.0, 7.5], [8.0, 9.0]])]),
    "tracks": lambda: sk.Tracks(np.array([[1.0, 0.0, 10.0, 12.0], [1.0, 1.0, 11.0, 13.5]])),
    "float": lambda: sk.Float(2.5, min=0.0, max=10.0),
    "integer": lambda: sk.Integer(7, min=0, max=10),
    "bool": lambda: sk.Bool(True),
    "string": lambda: sk.String("hello"),
    "choice": lambda: sk.Choice("b", items=["a", "b"]),
    "null": lambda: sk.Null(),
    "notification": lambda: sk.Notification("Done!", level="warning"),
    "any": lambda: sk.Any(
        {
            "array": np.arange(3),
            "none": None,
            "nested": [1, [2.5, "abcd"], (3, 4)],
            "scalar": np.int64(4),
            "text": "gray",
        }
    ),
}


@pytest.mark.parametrize("make_layer", LAYERS.values(), ids=LAYERS.keys())
def test_layer_roundtrip(make_layer, transport):
    layer = make_layer()

    _assert_same_layer(_roundtrip(layer, transport), layer)


def test_position_and_tile_meta_roundtrip(transport):
    tile_meta = TileMeta(
        tile_idx=2, n_tiles=4, first_tile=(False, True), last_tile=(True, False), overlap_px=(8, 8)
    )
    layer = sk.Image(np.zeros((6, 5), dtype=np.uint8), position=(10, 20), tile_meta=tile_meta)

    result = _roundtrip(layer, transport)

    assert tuple(result.position) == (10, 20)
    assert result.tile_meta.serialize() == tile_meta.serialize()


def test_meta_roundtrip(transport):
    """Arrays, Numpy scalars and nested structures in meta survive; plain strings stay strings."""
    layer = sk.Image(
        np.zeros((4, 4), dtype=np.uint8),
        contrast_limits=[np.float32(0.5), np.float32(2.0)],
        gamma=np.float64(0.25),
        features={"area": np.arange(3), "nested": {"weights": np.ones((2, 2))}},
        arrays=[np.arange(2), np.arange(3.0)],
        note="abcd",  # Valid base64: must not be decoded as an array
        colormap="gray",
    )

    result = _roundtrip(layer, transport)

    assert _normalize(result.meta) == _normalize(layer.meta)
    assert isinstance(result.meta["features"]["area"], np.ndarray)
    assert result.meta["note"] == "abcd"


def test_stack_roundtrip(transport):
    stack = sk.Stack(
        [
            sk.Image(np.zeros((4, 4), dtype=np.uint8), name="first"),
            sk.Points(np.array([[1.0, 2.0]]), name="second"),
            sk.Integer(3, name="third"),
        ]
    )

    result = StackSerializer.deserialize(transport(StackSerializer.serialize(stack)))

    assert [l.name for l in result] == ["first", "second", "third"]
    for result_layer, layer in zip(result, stack):
        _assert_same_layer(result_layer, layer)
