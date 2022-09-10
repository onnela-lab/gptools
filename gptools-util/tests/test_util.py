from gptools import util
import numpy as np
import pytest
import time


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
        time.sleep(0.2)
    time.sleep(0.1)
    assert 0.29 < timer.duration < 0.31
    out, _ = capsys.readouterr()
    assert out.startswith("timer test in 0.3")


def test_timer_errors():
    with util.Timer() as timer:
        pass
    assert timer.duration > 0
    with pytest.raises(RuntimeError, match="timers can only"), timer:
        pass

    timer = util.Timer()
    with pytest.raises(RuntimeError, match="timer has not yet"):
        timer.duration
