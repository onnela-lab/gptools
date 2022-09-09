from gptools.util import coordgrid, kernels
import itertools as it
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch as th


@pytest.fixture(params=[
    (kernels.ExpQuadKernel, (4, 0.2, 0.1), 3),
    (kernels.ExpQuadKernel, (1.2, 0.7, 0), 1),
    (kernels.ExpQuadKernel, (4, 0.2, 0.1, 2), 2),
    (kernels.ExpQuadKernel, (4, 0.2, 0.1, 2), 1),
    (kernels.ExpQuadKernel, (4, np.asarray([0.1, 0.15, 0.2]), 0.1, 2), 3),
])
def kernel_and_dim(request: pytest.FixtureRequest, use_torch: bool) -> tuple[kernels.Kernel, int]:
    cls, args, dim = request.param
    if use_torch:
        args = [th.as_tensor(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
    return cls(*args), dim


@pytest.mark.parametrize("shape", [(7,), (2, 3)])
def test_kernel(kernel_and_dim: tuple[kernels.Kernel, int], shape: tuple, use_torch: bool) -> None:
    kernel, p = kernel_and_dim
    X = th.randn(shape + (p,)) if use_torch else np.random.normal(0, 1, shape + (p,))
    if kernel.is_periodic:
        X = X % kernel.period
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


def test_periodic(kernel_and_dim: tuple[kernels.Kernel, int]) -> None:
    kernel, dim = kernel_and_dim
    if not kernel.is_periodic:
        pytest.skip("kernel is not periodic")
    # Sample some points from the domain.
    period = kernel.period * np.ones(dim)
    x = np.random.uniform(0, 1, (13, dim)) * period
    cov = kernel(x)
    # Ensure that translating by an integer period doesn't mess with the covariance.
    for delta in it.product(*([-1, 1, 2] for _ in range(dim))):
        y = x + delta * period
        other = kernel(x[..., :, None, :], y[..., None, :, :])
        np.testing.assert_allclose(cov, other)

    # For one and two dimensions, ensure that the Fourier transform has the correct structure.
    if dim > 2:
        return

    # We consider different shapes here to make sure we don't get unexpected behavior with edge
    # cases.
    for shape in it.product(*((5, 8) for _ in range(dim))):
        xs = coordgrid(*(np.linspace(0, width, size, False) for width, size in zip(period, shape)))
        assert xs.shape == (np.prod(shape), len(shape))

        # Evaluate the covariance with the origin, take the Fourier transform, and check that there
        # is no imaginary part.
        cov = kernel(xs)[0].reshape(shape)
        fftcov = np.fft.rfft(cov) if dim == 1 else np.fft.rfft2(cov)
        np.testing.assert_allclose(fftcov.imag, 0, atol=1e-9)
