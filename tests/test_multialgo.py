import imaging_server_kit as sk


@sk.algorithm
def algo1(x: int):
    return x**2

@sk.algorithm
def algo2(y: int):
    return y**3

def test_multialgo():
    multi = sk.combine([algo1, algo2])

    out1 = multi(algorithm="algo1", x=0)
    out2 = multi(algorithm="algo2", y=1)
    assert out1 == 0
    assert out2 == 1

    out1 = multi.run(algorithm="algo1", x=0)
    out2 = multi.run(algorithm="algo2", y=1)
    assert out1[0].data == 0
    assert out2[0].data == 1