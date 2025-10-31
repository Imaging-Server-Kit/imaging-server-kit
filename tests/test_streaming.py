import imaging_server_kit as sk


# Basic
@sk.algorithm
def basic_stream():
    yield sk.Notification("Success")

def test_basic_stream():
    assert basic_stream()[0] == "Success"
    assert basic_stream.run().read("Notification").data == "Success"


# Loop stream
@sk.algorithm
def loop_stream():
    for k in range(5):
        yield k

def test_loop_stream():
    assert loop_stream()[0] == 4
    assert loop_stream.run().read("Int").data == 4


# yield + return
@sk.algorithm
def stream_return():
    for k in range(3):
        yield sk.Integer(k)
    return sk.Notification("Finished")

def test_stream_return():
    out = stream_return()
    assert out[0] == 2
    assert out[1] == "Finished"

    out = stream_return.run()
    assert out[0].data == 2
    assert out[1].data == "Finished"


# multiple yield + multiple returns
@sk.algorithm
def multi_stream():
    for k in range(3):
        yield sk.Integer(k), sk.Float(0.5)
    # Integer here will override the previous integer (that's expected!)
    return sk.Notification("Finished"), sk.Integer(10)

def test_multi_stream():
    out = multi_stream()
    assert out[0] == 10
    assert out[1] == 0.5
    assert out[2] == "Finished"

    out = multi_stream.run()
    assert out[0].data == 10
    assert out[1].data == 0.5
    assert out[2].data == "Finished"


### Streamed image

### Streamed mask

### Streamed points

### Streamed vectors

### Streamed boxes
