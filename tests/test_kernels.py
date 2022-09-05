from gptools import kernels
from gptools.missing_module import MissingModule
import numpy as np
import pytest
try:
    import torch as th
except ModuleNotFoundError as ex:
    th = MissingModule(ex)


@pytest.mark.parametrize("kernel", [
    kernels.ExpQuadKernel(4, 0.2, .1),
    kernels.ExpQuadKernel(1.2, 0.7, 0),
])
@pytest.mark.parametrize("p", [1, 5])
@pytest.mark.parametrize("shape", [(7,), (2, 3)])
@pytest.mark.parametrize("torch", [False, True])
def test_kernel(kernel: kernels.ExpQuadKernel, p: int, shape: tuple, torch: bool) -> None:
    if torch:
        try:
            X = th.randn(shape + (p,))
        except ModuleNotFoundError:
            pytest.skip("torch is not installed")
    else:
        X = np.random.normal(0, 1, shape + (p,))
    cov = kernel(X)
    # Check the shape and that the kernel is positive definite.
    *batch_shape, n = shape
    assert cov.shape == tuple(batch_shape) + (n, n)
    if kernel.epsilon:
        np.linalg.cholesky(cov)
