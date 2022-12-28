import multiprocessing
import numbers
import psutil
from queue import Empty
import traceback
from typing import Any, Callable


def _wrapper(queue, target, *args, **kwargs):  # pragma: no cover
    """
    Wrapper to execute a function in a subprocess.
    """
    try:
        result = target(*args, **kwargs)
        success = True
    except Exception as ex:
        result = (ex, traceback.format_exc())
        success = False
    queue.put_nowait((success, result))


def call_with_timeout(timeout: float, target: Callable, *args, **kwargs) -> Any:
    """
    Call a target with a timeout and return its result.

    Args:
        timeout: Number of seconds to wait for a result.
        target: Function to call.
        *args: Positional arguments passed to `target`.
        **kwargs: Keyword arguments passed to `target`.

    Returns:
        result: Return value of `target`.

    Raises:
        TimeoutError: If the target does not complete within the timeout.
    """
    if not isinstance(timeout, numbers.Number) or timeout <= 0:
        raise ValueError("timeout must be a positive number")
    if not callable(target):
        raise TypeError("target must be callable")
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_wrapper, args=(queue, target, *args), kwargs=kwargs,
                                      daemon=True)
    process.start()

    try:
        success, result = queue.get(timeout=timeout)
    except Empty:
        raise TimeoutError(f"failed to fetch result after {timeout} seconds")
    finally:
        if process.is_alive():
            # Kill the process and all its children (https://stackoverflow.com/a/4229404/1150961).
            children = psutil.Process(process.pid).children(recursive=True)
            for child in children:
                child.terminate()  # pragma: no cover
            _, still_alive = psutil.wait_procs(children, timeout=3)
            if still_alive:
                raise RuntimeError("some processes are still alive")  # pragma: no cover
            process.terminate()
        process.join(timeout=5)
    if not success:
        ex, tb = result
        raise RuntimeError(tb) from ex
    return result
