from __future__ import annotations
import functools as ft
import numpy as np
import os
import time
import typing


# This tricks pylance into thinking that the imports *could* happen for type checking.
FALSE = os.environ.get("f1571823-5638-473a-adb5-6f5efb0cb773")
if FALSE:
    import torch as th


ArrayOrTensor = typing.Union[np.ndarray, "th.Tensor"]


class ArrayOrTensorDispatch:
    """
    Call the equivalent numpy or torch function based on the value of arguments.
    """
    def __getattr__(self, name: str) -> typing.Callable:
        return ft.partial(self, name)

    def __call__(self, name, x: ArrayOrTensor, *args, **kwargs) -> typing.Any:
        return getattr(self[x], name)(x, *args, **kwargs)

    def __getitem__(self, x: ArrayOrTensor) -> typing.Any:
        try:
            import torch as th
            if isinstance(x, th.Tensor):
                return th
        except ModuleNotFoundError:  # pragma: no cover
            pass
        return np


def coordgrid(*xs: typing.Iterable[np.ndarray], ravel: bool = True,
              indexing: typing.Literal["ij", "xy"] = "ij") -> np.ndarray:
    """
    Obtain coordinates for all grid points induced by `xs`.

    Args:
        xs: Coordinates to construct the grid.
        ravel: Whether to reshape the leading dimensions.
        indexing: Whether to use Cartesian `xy` or matrix `ij` indexing (defaults to `ij`).

    Returns:
        coord: Coordinates for all grid points with shape `(len(xs[0]), ..., len(xs[p - 1]), p)` if
            `ravel` is `False`, where `p = len(xs)` is the number of dimensions. If `ravel` is
            `True`, the shape is `(len(xs[0]) * ... * len(xs[p - 1]), p)`.
    """
    # Stack the coordinate matrices and move the coordinate dimension to the back.
    coords = np.moveaxis(np.stack(np.meshgrid(*xs, indexing=indexing)), 0, -1)
    if not ravel:
        return coords
    return coords.reshape((-1, len(xs)))


class Timer:
    """
    Time the duration of code execution in a context.

    Args:
        message: Message to print when the context becomes inactive.
    """
    def __init__(self, message: str = None):
        self.start = None
        self.end = None
        self.message = message

    def __enter__(self) -> Timer:
        if self.start is not None:
            raise RuntimeError("timers can only be used once")
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end = time.time()
        if self.message:
            print(f"{self.message} in {self.duration:.3f} seconds")

    @property
    def duration(self) -> float:
        """
        The duration between the start and end of the context. Returns the time since the start if
        the context is still active.
        """
        if self.start is None:
            raise RuntimeError("timer has not yet been started")
        end = self.end or time.time()
        return end - self.start

    def __repr__(self) -> str:
        try:
            return f"Timer(duration={self.duration:.3f})"
        except RuntimeError:
            return "Timer(not started)"
