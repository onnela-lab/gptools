from __future__ import annotations
import functools as ft
import inspect
import numpy as np
import os
import time
from typing import Any, Callable, Iterable, Literal, Optional, Union


# This tricks pylance into thinking that the imports *could* happen for type checking.
FALSE = os.environ.get("f1571823-5638-473a-adb5-6f5efb0cb773")
if FALSE:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    import torch as th


ArrayOrTensor = Union[np.ndarray, "th.Tensor"]
OptionalArrayOrTensor = Optional[ArrayOrTensor]


class ArrayOrTensorDispatch:
    """
    Call the equivalent numpy or torch function based on the value of arguments.
    """
    def __getattr__(self, name: str) -> Callable:
        return ft.partial(self, name)

    def __call__(self, name, *args, **kwargs) -> Any:
        module = self[args + tuple(kwargs.values())]
        return getattr(module, name)(*args, **kwargs)

    def __getitem__(self, x: ArrayOrTensor) -> Any:
        if self.is_tensor(*x) if isinstance(x, tuple) else self.is_tensor(x):
            import torch
            return torch
        else:
            return np

    def is_tensor(self, *xs: Iterable[ArrayOrTensor]) -> bool:
        """
        Check if objects are tensors.

        Args:
            xs: Objects to check.

        Returns:
            is_tensor: `True` if `x` is a tensor; `False` otherwise.

        Raises:
            ValueError: If the objects are a mixture of torch tensors and numpy arrays.
        """
        #
        try:
            import torch as th
        except ModuleNotFoundError:
            return False
        # None of the arguments are tensors.
        if not any(isinstance(x, th.Tensor) for x in xs):
            return False
        # At least one is a tensor. If another is an array, we're in trouble.
        if any(isinstance(x, np.ndarray) for x in xs):
            raise ValueError("arguments are a mixture of torch tensors and numpy arrays")
        return True

    def get_complex_dtype(self, x: ArrayOrTensor) -> Any:
        """
        Get the complex dtype matching `x` in precision.
        """
        return (1j * self[x].empty(0, dtype=x.dtype)).dtype

    @ft.wraps(np.concatenate)
    def concatenate(self, arrays: Iterable[ArrayOrTensor],
                    axis: Optional[Union[int, tuple[int]]] = None) -> ArrayOrTensor:
        if self.is_tensor(*arrays):
            import torch as th
            return th.concat(arrays, dim=axis)
        else:
            return np.concatenate(arrays, axis=axis)


dispatch = ArrayOrTensorDispatch()


def coordgrid(*xs: Iterable[np.ndarray], ravel: bool = True, indexing: Literal["ij", "xy"] = "ij") \
        -> np.ndarray:
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


class mutually_exclusive_kwargs:
    """
    Ensure a function receives mutually exclusive keyword arguments.

    Args:
        keys: Sequence of string or string tuple keys. Tuples indicate that none or all of the keys
            must be given together.
    """
    def __init__(self, *keys) -> None:
        self.keys = keys

    def __call__(self, func: Callable) -> Callable:
        @ft.wraps(func)
        def _wrapper(*args, **kwargs) -> Any:
            # Assemble the values from the keys.
            given_key = None
            for key in self.keys:
                if isinstance(key, str):
                    given = kwargs.get(key) is not None
                else:
                    given = {x: kwargs.get(x) is not None for x in key}
                    if all(given.values()) != any(given.values()):
                        raise ValueError(f"some but not all of {key} are given: {given}")
                    given = all(given.values())
                if given and given_key:
                    raise ValueError(f"`{key}` and `{given_key}` are both given")
                if given:
                    given_key = key
            if not given_key:
                raise ValueError(f"expected exactly one of {self.keys} to be given but got "
                                 f"{kwargs}")
            if "given" in list(inspect.signature(func).parameters):
                kwargs["given"] = given_key
            return func(*args, **kwargs)
        return _wrapper


def encode_one_hot(z: ArrayOrTensor, p: Optional[int] = None) -> ArrayOrTensor:
    """
    Encode a vector of integers as a one-hot matrix.

    Args:
        z: Array of integers.
        p: Number of classes (defaults to `max(z) + 1`).

    Returns:
        one_hot: One-hot encoded values.
    """
    p = p or z.max() + 1
    n = z.shape[0]
    result = dispatch[z].zeros((n, p))
    result[dispatch[z].arange(n), z] = 1
    return result


def match_colorbar(cb: "Colorbar", ax: Optional["Axes"] = None) -> tuple[float]:
    """
    Match the size of the colorbar with the size of the axes.

    Args:
        ax: Axes from which the colorbar "stole" space.
        cb: Colorbar to match to `ax`.

    Returns:
        pos: New position of the colorbar axes.
    """
    from matplotlib import pyplot as plt

    ax = ax or plt.gca()
    bbox = ax.get_position()
    cb_bbox = cb.ax.get_position()
    cb.ax.set_aspect("auto")
    if cb.orientation == "vertical":
        # Update bottom and height.
        left = cb_bbox.xmin
        width = cb_bbox.width
        bottom = bbox.ymin
        height = bbox.height
    else:
        # Update left and width.
        left = bbox.xmin
        width = bbox.width
        bottom = cb_bbox.ymin
        height = cb_bbox.height
    pos = (left, bottom, width, height)
    cb.ax.set_position(pos)
    return pos
