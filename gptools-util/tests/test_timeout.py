from functools import partial
from gptools.util import Timer
from gptools.util.timeout import call_with_timeout
import pytest
import time


def target(wait: float, fail: bool) -> float:
    if fail:
        raise NotImplementedError("that didn't work")
    time.sleep(wait)
    return wait


@pytest.mark.parametrize("timeout, wait, fail", [
    (1, 0.5, False),
    (1, 0.5, True),
    (0.5, 1, False),
    (0.5, 1, True),
])
def test_timeout(timeout: float, wait: float, fail: bool) -> None:
    tol = 0.3  # Need some tolerance because the child takes some time to start.
    p = partial(call_with_timeout, timeout, target, wait, fail=fail)
    if fail:
        with Timer() as timer, pytest.raises(RuntimeError):
            p()
        assert timer.duration < tol
        return
    if timeout < wait:
        with Timer() as timer, pytest.raises(TimeoutError):
            p()
        assert timer.duration > timeout
        return
    with Timer() as timer:
        assert p() == wait
    assert timer.duration < wait + tol


def test_timeout_negative_timeout():
    with pytest.raises(ValueError):
        call_with_timeout(-1, None)


def test_timeout_not_a_callable():
    with pytest.raises(TypeError):
        call_with_timeout(3, None)
