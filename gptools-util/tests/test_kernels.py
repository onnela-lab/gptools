from gptools.util import kernels
from gptools.util.testing import KernelConfiguration
import itertools as it
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch as th


@pytest.mark.parametrize("shape", [(7,), (2, 3)])
def test_kernel(kernel_configuration: KernelConfiguration, shape: tuple, use_torch: bool) -> None:
    kernel = kernel_configuration()
    X = kernel_configuration.sample_locations(shape)
    cov = kernel(X)
    # Check the shape and that the kernel is positive definite if there is nugget variance.
    *batch_shape, n = shape
    assert cov.shape == tuple(batch_shape) + (n, n)
    if kernel.epsilon:
        np.linalg.cholesky(cov)


@pytest.mark.parametrize("p", [1, 3, 5])
def test_evaluate_squared_distance(p: int, use_torch: bool) -> None:
    shape = (57, p)
    X = th.randn(shape) if use_torch else np.random.normal(0, 1, shape)
    np.testing.assert_allclose(kernels.evaluate_squared_distance(X), cdist(X, X) ** 2)


def test_periodic(kernel_configuration: KernelConfiguration):
    kernel = kernel_configuration()
    if not kernel.is_periodic:
        pytest.skip("kernel is not periodic")
    # Sample some points from the domain.
    X = kernel_configuration.sample_locations((13,))
    _, dim = X.shape
    cov = kernel(X)
    # Ensure that translating by an integer period doesn't mess with the covariance.
    for delta in it.product(*([-1, 1, 2] for _ in range(dim))):
        Y = X + delta * np.asarray(kernel.period)
        other = kernel(X[..., :, None, :], Y[..., None, :, :])
        np.testing.assert_allclose(cov, other)

    # For one and two dimensions, ensure that the Fourier transform has the correct structure.
    if dim > 2:
        return

    # We consider different shapes here to make sure we don't get unexpected behavior with edge
    # cases.
    for shape in it.product(*((5, 8) for _ in range(dim))):
        xs = kernel_configuration.coordgrid(shape)
        assert xs.shape == (np.prod(shape), len(shape))

        # Evaluate the covariance with the origin, take the Fourier transform, and check that there
        # is no imaginary part.
        cov = kernel(xs)[0].reshape(shape)
        fftcov = np.fft.rfft(cov) if dim == 1 else np.fft.rfft2(cov)
        np.testing.assert_allclose(fftcov.imag, 0, atol=1e-9)
