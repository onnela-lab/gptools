from gptools.util import coordgrid, kernels
from gptools.util.testing import KernelConfiguration
import itertools as it
import mpmath
import numpy as np
import pytest
from scipy.spatial.distance import cdist
import torch as th


@pytest.mark.parametrize("q", [0.1, 0.8])
def test_jtheta(q: float) -> None:
    z = np.linspace(0, 1, 7, endpoint=False)
    actual = kernels.jtheta(z, q)
    desired = np.vectorize(mpmath.jtheta)(3, np.pi * z, q).astype(float)
    np.testing.assert_allclose(actual, desired)


@pytest.mark.parametrize("nz", [5, 6])
@pytest.mark.parametrize("q", [0.1, 0.8])
def test_jtheta_rfft(nz: int, q: float) -> None:
    jtheta = kernels.jtheta(np.linspace(0, 1, nz, endpoint=False), q)
    actual = kernels.jtheta_rfft(nz, q)
    desired = np.fft.rfft(jtheta)
    np.testing.assert_allclose(actual, desired)


@pytest.mark.parametrize("shape", [(7,), (2, 3)])
def test_kernel(kernel_configuration: KernelConfiguration, shape: tuple, use_torch: bool) -> None:
    kernel = kernel_configuration()
    X = kernel_configuration.sample_locations(shape)
    cov = (kernel + kernels.DiagonalKernel(1e-3, kernel.period)).evaluate(X)
    # Check the shape and that the kernel is positive definite if there is nugget variance.
    *batch_shape, n = shape
    assert cov.shape == tuple(batch_shape) + (n, n)
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
    cov = kernel.evaluate(X)
    # Ensure that translating by an integer period doesn't mess with the covariance.
    for delta in it.product(*([-1, 1, 2] for _ in range(dim))):
        Y = X + delta * np.asarray(kernel.period)
        other = kernel.evaluate(X[..., :, None, :], Y[..., None, :, :])
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
        cov = kernel.evaluate(xs)[0].reshape(shape)
        fftcov = np.fft.rfft(cov) if dim == 1 else np.fft.rfft2(cov)
        np.testing.assert_allclose(fftcov.imag, 0, atol=1e-9)


def test_kernel_composition():
    a = kernels.ExpQuadKernel(2, 0.5)
    b = kernels.DiagonalKernel()
    kernel = a + 1 + b
    x = np.random.normal(0, 1, (100, 2))
    np.testing.assert_allclose(kernel.evaluate(x), a.evaluate(x) + 1 + b.evaluate(x))


def test_kernel_composition_period():
    with pytest.raises(ValueError):
        kernels.DiagonalKernel(period=1) + kernels.DiagonalKernel()
    with pytest.raises(ValueError):
        kernels.DiagonalKernel(period=1) + kernels.DiagonalKernel(period=2)


def test_diagonal_kernel():
    kernel = kernels.DiagonalKernel()
    x = np.random.normal(0, 1, (10, 2))
    np.testing.assert_allclose(kernel.evaluate(x), np.eye(10))
    with pytest.raises(ValueError):
        kernel.evaluate(x, x)


@pytest.mark.parametrize("shape", [(5,), (6,), (5, 7), (5, 6), (6, 5), (6, 8)])
def test_heat_kernel(shape: int) -> None:
    *head, tail = shape
    ndim = len(shape)
    # Use a large number of terms to evaluate the kernel.
    if ndim == 1:
        kernel = kernels.HeatKernel(1.2, 0.1, 3, tail // 2 + 1)
    elif ndim == 2:
        kernel = kernels.HeatKernel(0.9, np.asarray([0.2, 0.3]), np.asarray([2.1, 2.3]))
    else:
        raise ValueError
    xs = coordgrid(*[np.linspace(0, period, n, endpoint=False) for n, period in
                     zip(shape, kernel.period * np.ones(ndim))])
    cov = kernel.evaluate(xs)[0].reshape(shape)
    if ndim == 1:
        rfft = np.fft.rfft(cov)
    elif ndim == 2:
        rfft = np.fft.rfft2(cov)
    else:
        raise ValueError
    assert rfft.shape == (*head, tail // 2 + 1,)
    np.testing.assert_allclose(rfft.imag, 0, atol=1e-9)
    rfft = rfft.real
    predicted = kernel.evaluate_rfft(shape)
    np.testing.assert_allclose(rfft, predicted, atol=1e-9)


@pytest.mark.parametrize("num_terms", [None, 7, np.arange(4)])
def test_heat_kernel_num_terms(num_terms) -> None:
    kernel = kernels.HeatKernel(1, .5, 1, num_terms)
    assert kernel.num_terms >= 1


def test_heat_kernel_without_period() -> None:
    with pytest.raises(ValueError):
        kernels.HeatKernel(1.2, 0.5, None)
