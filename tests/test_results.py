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

    results = sk.Results()
    assert len(results.layers) == 0
    out = minimal.run(results=results)
    assert out is results
    assert len(results.layers) == 1
    assert results.layers[0].data == 1

    out = override.run(results=results)
    assert len(results.layers) == 1
    assert results.layers[0].data == 2

    out = extra.run(results=results)
    assert len(results.layers) == 2