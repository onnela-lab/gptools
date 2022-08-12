from graph_gaussian_process import kernels
import numpy as np
import pytest
import torch as th
import typing


@pytest.mark.parametrize("kernel", [
    kernels.ExpQuadKernel(4, 0.2, .1),
    kernels.ExpQuadKernel(1.2, 0.7, 0),
])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("shape", [(7,), (2, 3)])
@pytest.mark.parametrize("torch", [False, True])
def test_kernel(kernel: typing.Callable, p: int, shape: tuple, torch: bool) -> None:
    X = th.randn(shape + (p,)) if torch else np.random.normal(0, 1, shape + (p,))
    cov = kernel(X)
    # Check the shape and that the kernel is positive definite.
    *batch_shape, n = shape
    assert cov.shape == tuple(batch_shape) + (n, n)
    np.linalg.cholesky(cov)
