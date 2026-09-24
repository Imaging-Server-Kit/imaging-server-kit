import numpy as np
import pytest

import imaging_server_kit as sk


def test_domain_repr():
    assert repr(sk.Domain()) == "Domain(undefined)"
    domain = sk.Domain(size=(512, 256), position=(0, 100))
    assert repr(domain) == "Domain(position=(0, 100), size=(512, 256))"
    assert str(domain) == repr(domain)


def test_tile_meta_repr():
    assert repr(sk.TileMeta()) == "TileMeta(untiled)"
    tile_meta = sk.TileMeta(
        tile_idx=3,
        n_tiles=16,
        overlap_px=(8, 8),
        first_tile=(True, False),
        last_tile=(False, False),
    )
    assert repr(tile_meta) == (
        "TileMeta(tile=3/16, overlap_px=(8, 8), first=(T, F), last=(F, F))"
    )


@pytest.mark.parametrize(
    "layer, expected",
    [
        (sk.Image(data=np.zeros((32, 16), "uint8")), "<Image 'Image' uint8 (32, 16)>"),
        (
            sk.Image(data=np.zeros((8, 8, 3), "uint8"), rgb=True),
            "<Image 'Image' uint8 (8, 8, 3), rgb>",
        ),
        (sk.Image(required=False), "<Image 'Image' empty>"),
        (sk.Points(data=np.ones((5, 2))), "<Points 'Points' 5 points, 2D>"),
        (sk.Float(name="Sigma", data=1.5), "<Float 'Sigma' = 1.5>"),
        (
            sk.Float(name="Sigma", data=1.5, min=0, max=10),
            "<Float 'Sigma' = 1.5 in [0, 10]>",
        ),
        (sk.Integer(name="Size", data=3), "<Integer 'Size' = 3>"),
        (sk.Bool(name="Invert", data=False), "<Bool 'Invert' = False>"),
        (
            sk.Choice(name="Mode", data="reflect", items=["reflect", "constant"]),
            "<Choice 'Mode' = 'reflect' of {reflect, constant}>",
        ),
        (sk.Null(), "<Null 'None'>"),
    ],
)
def test_layer_repr(layer, expected):
    assert repr(layer) == expected


def test_mask_repr_counts_labels_without_background():
    mask = np.zeros((16, 16), "int32")
    mask[:4, :4] = 1
    mask[8:, 8:] = 2
    assert repr(sk.Mask(data=mask)) == "<Mask 'Mask' int32 (16, 16), 2 labels>"


def test_layer_repr_shows_position_and_tile():
    layer = sk.Image(data=np.zeros((32, 32), "uint8"))[10:20, 5:15]
    assert "at (10, 5)" in repr(layer)
    layer.tile_meta = sk.TileMeta(tile_idx=1, n_tiles=4)
    assert "tile 1/4" in repr(layer)


def test_long_string_is_truncated():
    layer = sk.String(data="x" * 200)
    assert len(repr(layer)) < 80


def test_stack_repr_and_str():
    assert repr(sk.Stack()) == "<Stack 0 layers>"
    assert str(sk.Stack()) == "Stack | empty"

    stack = sk.Stack(
        [
            sk.Image(data=np.zeros((64, 32), "uint8")),
            sk.Float(name="Threshold", data=0.5),
        ]
    )
    assert repr(stack) == "<Stack 2 layers, extent [0:64, 0:32]>"

    lines = str(stack).splitlines()
    assert lines[0] == "Stack | 2 layers | extent [0:64, 0:32]"
    assert "uint8 (64, 32)" in lines[2]
    assert "Threshold" in lines[3] and "= 0.5" in lines[3]


def test_large_stack_str_is_truncated():
    stack = sk.Stack([sk.Float(data=float(i)) for i in range(50)])
    lines = str(stack).splitlines()
    assert len(lines) < 20
    assert any(line.strip() == "..." for line in lines)
