from gptools import util
import numpy as np
import pytest
import time
import torch as th


@pytest.mark.parametrize("shape", [(3,), (4, 5)])
@pytest.mark.parametrize("ravel", [False, True])
def test_coord_grid(shape: tuple[int], ravel: bool) -> None:
    xs = [np.arange(p) for p in shape]
    coords = util.coordgrid(*xs, ravel=ravel)
    if ravel:
        assert coords.shape == (np.prod(shape), len(shape),)
    else:
        assert coords.shape == shape + (len(shape),)


def test_timer(capsys: pytest.CaptureFixture):
    with util.Timer("timer test") as timer:
        time.sleep(0.1)
        assert 0.09 < timer.duration < 0.11
        time.sleep(0.05)
    time.sleep(0.1)
    assert 0.14 < timer.duration < 0.16
    out, _ = capsys.readouterr()
    assert out.startswith("timer test in 0.15")


def test_timer_errors():
    with util.Timer() as timer:
        time.sleep(1e-6)
    assert timer.duration > 0
    with pytest.raises(RuntimeError, match="timers can only"), timer:
        pass

    timer = util.Timer()
    with pytest.raises(RuntimeError, match="timer has not yet"):
        timer.duration


def test_dispatch():
    dispatch = util.ArrayOrTensorDispatch()

    tensor = th.empty(7)
    assert dispatch[tensor] is th
    assert dispatch[tensor, 0.1] is th
    assert isinstance(dispatch.add(tensor, tensor), th.Tensor)

    array = np.empty(3)
    assert dispatch[array] is np
    assert dispatch[array, 0.2] is np
    assert isinstance(dispatch.add(array, array), np.ndarray)

    with pytest.raises(ValueError):
        dispatch[tensor, array]
    with pytest.raises(ValueError):
        dispatch.add(array, tensor)


@pytest.mark.parametrize("float_, complex_", [
    (np.float32, np.complex64),
    (np.float64, np.complex128),
    (th.float32, th.complex64),
    (th.float64, th.complex128),
])
def test_get_complex_dtype(float_, complex_) -> None:
    dispatch = util.ArrayOrTensorDispatch()
    x = (th if float_.__module__ == "torch" else np).ones(3, dtype=float_)
    assert dispatch.get_complex_dtype(x) == complex_
