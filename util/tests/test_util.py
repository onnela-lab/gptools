from gptools import util
from matplotlib import pyplot as plt
import numpy as np
import pytest
import re
import time
from typing import Tuple


@pytest.mark.parametrize("shape", [(3,), (4, 5)])
@pytest.mark.parametrize("ravel", [False, True])
def test_coord_grid(shape: Tuple[int], ravel: bool) -> None:
    xs = [np.arange(p) for p in shape]
    coords = util.coordgrid(*xs, ravel=ravel)
    if ravel:
        assert coords.shape == (
            np.prod(shape),
            len(shape),
        )
    else:
        assert coords.shape == shape + (len(shape),)


def test_timer(capsys: pytest.CaptureFixture):
    with util.Timer("timer test") as timer:
        time.sleep(0.1)
        assert 0.09 < timer.duration < 0.11
        time.sleep(0.05)
    time.sleep(0.1)
    assert 0.14 < timer.duration < 0.17
    out, _ = capsys.readouterr()
    assert re.match(r"timer test in 0\.1[56]", out)


def test_timer_errors():
    with util.Timer() as timer:
        time.sleep(1e-6)
    assert timer.duration > 0
    with pytest.raises(RuntimeError, match="timers can only"), timer:
        pass

    timer = util.Timer()
    with pytest.raises(RuntimeError, match="timer has not yet"):
        timer.duration


def test_mutually_exclusive_kwargs() -> None:
    @util.mutually_exclusive_kwargs("a", ("b", "c"))
    def _func(a=None, b=None, c=None, given=None) -> None:
        return given

    assert _func(a=3) == "a"
    assert _func(b=2, c=1) == ("b", "c")
    with pytest.raises(ValueError):
        _func()
    with pytest.raises(ValueError):
        _func(a=3, b=2)
    with pytest.raises(ValueError):
        _func(b=2)
    with pytest.raises(ValueError):
        _func(a=3, b=2, c=1)


def test_encode_one_hot() -> None:
    x = np.random.randint(5, size=1000)
    y = util.encode_one_hot(x)
    assert y.shape == (x.shape[0], x.max() + 1)
    np.testing.assert_array_equal(x, np.argmax(y, axis=1))


@pytest.mark.parametrize("location", ["top", "right"])
def test_match_colorbar(location: str) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(np.random.normal(0, 1, (5, 5)))
    cb = fig.colorbar(im, ax=ax, location=location)
    util.match_colorbar(cb)
