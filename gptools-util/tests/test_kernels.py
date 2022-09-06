from gptools.util import kernels
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch as th


@pytest.mark.parametrize("kernel", [
    kernels.ExpQuadKernel(4, 0.2, .1),
    kernels.ExpQuadKernel(1.2, 0.7, 0),
    kernels.ExpQuadKernel(4, 0.2, .1, 2),
    kernels.ExpQuadKernel(1.2, 0.7, 1.5),
])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("shape", [(7,), (2, 3)])
def test_kernel(kernel: kernels.ExpQuadKernel, p: int, shape: tuple, use_torch: bool) -> None:
    X = th.randn(shape + (p,)) if use_torch else np.random.normal(0, 1, shape + (p,))
    cov = kernel(X)
    # Check the shape and that the kernel is positive definite.
    *batch_shape, n = shape
    assert cov.shape == tuple(batch_shape) + (n, n)
    if kernel.epsilon:
        np.linalg.cholesky(cov)


@pytest.mark.parametrize("p", [1, 3, 5])
def test_evaluate_squared_distance(p: int, use_torch: bool) -> None:
    shape = (57, p)
    X = th.randn(shape) if use_torch else np.random.normal(0, 1, shape)
    np.testing.assert_allclose(kernels.evaluate_squared_distance(X), cdist(X, X) ** 2)
