from functools import partial
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
    p = partial(call_with_timeout, timeout, target, wait, fail=fail)
    if fail:
        with pytest.raises(RuntimeError):
            p()
        return
    if timeout < wait:
        with pytest.raises(TimeoutError):
            p()
        return
    assert p() == wait


def test_timeout_negative_timeout():
    with pytest.raises(ValueError):
        call_with_timeout(-1, None)


def test_timeout_not_a_callable():
    with pytest.raises(TypeError):
        call_with_timeout(3, None)
