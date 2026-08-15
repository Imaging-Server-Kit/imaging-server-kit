import imaging_server_kit as sk


@sk.algorithm
def minimal():
    return 1


@sk.algorithm
def override():
    return 2


@sk.algorithm
def extra():
    return sk.Integer(3, name="Other")


def test_rediretcion():
    out = minimal.run()
    assert out[0].data == 1

    stack = sk.Stack()
    assert len(stack.layers) == 0
    out = minimal.run(stack=stack)
    assert out is stack
    assert len(stack.layers) == 1
    assert stack.layers[0].data == 1

    out = override.run(stack=stack)
    assert len(stack.layers) == 1
    assert stack.layers[0].data == 2

    out = extra.run(stack=stack)
    assert len(stack.layers) == 2
