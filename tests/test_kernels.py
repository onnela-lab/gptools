from graph_gaussian_process import kernels
import numpy as np
import pytest
import typing


@pytest.mark.parametrize("kernel", [
    kernels.ExpQuadKernel(4, 0.2, .1),
])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("shape", [(7,), (2, 3)])
def test_kernel(kernel: typing.Callable, p: int, shape: tuple) -> None:
    X = np.random.normal(0, 1, shape + (p,))
    cov = kernel(X)
    # Check the shape and that the kernel is positive definite.
    *batch_shape, n = shape
    assert cov.shape == tuple(batch_shape) + (n, n)
    np.linalg.cholesky(cov)
