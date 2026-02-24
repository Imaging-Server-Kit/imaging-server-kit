import numpy as np
import pytest
import imaging_server_kit as sk


# Empty algorithm
@sk.algorithm()
def empty():
    pass

def test_empty():
    out = empty()
    assert out is None
    out = empty.run()
    assert out[0].data is None


# Minimal output
@sk.algorithm()
def minimal():
    return sk.Integer(1)

def test_minimal():
    out = minimal()
    assert out == 1

    out = minimal.run()
    assert len(out) == 1
    assert out[0].data == 1


# Multi-output
@sk.algorithm()
def multioutput():
    return sk.String("Output 1"), sk.String("Output 2")

def test_multioutput():
    out1, out2 = multioutput()
    assert out1 == "Output 1"
    assert out2 == "Output 2"

    out = multioutput.run()
    assert len(out) == 2
    assert out[0].data == "Output 1"
    assert out[1].data == "Output 2"
    assert out[1].name.endswith("-01")


# Non-sk outputs
@sk.algorithm
def minimal_int():
    return 1

def test_minimal_int():
    out = minimal_int()
    assert out == 1
    out = minimal_int.run()
    assert out[0].data == 1
    

# Using sk.algorithm(func)
def minimal_float():
    return 3.14

def test_minimal_float():
    fl = sk.algorithm(minimal_float)
    assert fl() == 3.14
    assert fl.run()[0].data == 3.14


# AlgorithmServer + type annotations
def multiply(x: float, y: float):
    return sk.Float(x*y)

def test_multiply():
    algo = sk.Algorithm(multiply)
    out = algo.run(3, 4)
    assert out[0].data == 12


# Single image + first-only
@sk.algorithm(
    samples=[{"image": np.random.random((10, 10))}]
)
def sample_image(image):
    pass

def test_sample_image():
    sample = sample_image.get_sample()
    image = sample[0].data
    assert isinstance(image, np.ndarray)


# Basic parameters
@sk.algorithm(parameters={"width": sk.Float(), "length": sk.Integer()})
def algo_with_params(width, length):
    return sk.String(f"Area: {width*length}")

def test_algo_with_params():
    out = algo_with_params(3, 4)
    assert out.endswith("12")
    out = algo_with_params.run(3, 4)
    assert out[0].data.endswith("12")


# With defaults
@sk.algorithm(
    parameters={
        "y": sk.Integer(default=3),
        "x": sk.Integer(default=4),
    },
)
def algo_with_defaults(x, y):
    return sk.Integer(x), sk.Integer(y)

def test_algo_with_defaults():
    assert algo_with_defaults(5) == [5, 3]
    assert algo_with_defaults() == [4, 3]
    assert algo_with_defaults(1, 2) == [1, 2]
    assert algo_with_defaults(2, y=1) == [2, 1]
    assert algo_with_defaults.run(2, y=1)[0].data == 2
    assert algo_with_defaults.run(2, y=1)[1].data == 1


# Type hints, partial parameters annotation
@sk.algorithm
def type_hinted(x: int, y: float):
    return x + y

def test_type_hinted():
    assert type_hinted.run(x=5, y=3.14)[0].data == 8.14


# Partial parameter annotation
@sk.algorithm(parameters={"y": sk.Float()})
def type_hinted_partial(x: int, y):
    return x + y

def test_type_hinted_partial():
    assert type_hinted_partial.run(x=5, y=3.14)[0].data == 8.14


# Default parameters
def type_hinted_defaults(x, y=3.14):
    return x + y

sk_type_hinted = sk.algorithm(type_hinted_defaults, {"x": sk.Integer(default=4)})

def test_type_hinted_defaults():
    assert sk_type_hinted.run(5, 3.14)[0].data == 8.14


# Setting required=False and passing no data allows these data to be None
@sk.algorithm
def int_required_false(val=sk.Integer(required=False)):
    return sk.Bool(val is None)

def test_required_int_false():
    results = int_required_false.run()
    assert results[0].data is None


# Setting required=True and passing no data sets the data to 0 (default integer)
@sk.algorithm
def int_required_true(val=sk.Integer(required=True)):
    return sk.Bool(val is None)

def test_required_int_false():
    results = int_required_true.run()
    assert results[0].data == 0