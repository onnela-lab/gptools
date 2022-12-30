from gptools.util import coordgrid, kernels
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
        # Ensure the rfft cannot be evaluated and skip the rest.
        with pytest.raises(NotImplementedError):
            kernel.evaluate_rfft(tuple(range(13, 13 + len(kernel_configuration.dims))))
        return

    # Sample some points from the domain.
    X = kernel_configuration.sample_locations((13,))
    _, dim = X.shape
    cov = kernel.evaluate(X)
    # Ensure that translating by an integer period doesn't mess with the covariance.
    for delta in it.product(*([-1, 1, 2] for _ in range(dim))):
        Y = X + delta * np.asarray(kernel.period)
        other = kernel.evaluate(X[..., :, None, :], Y[..., None, :, :])
        np.testing.assert_allclose(cov, other, atol=1e-12)

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

        # We may want to check that the numerical and theoretic FFT match, but this requires a more
        # "proper" implementation of the periodic kernels involving infinite sums (see
        # https://github.com/tillahoffmann/gptools/issues/59 for details). For now, let's verify we
        # have a positive-definite kernel.
        np.testing.assert_array_less(-1e-12, kernel.evaluate_rfft(shape))


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


@pytest.mark.parametrize("size", [500, (501,), (500, 502), (500, 501), (501, 500), (501, 503)])
def test_periodic_exp_quad_rfft(size: int) -> None:
    shape = [size] if isinstance(size, int) else size
    *head, tail = shape
    ndim = len(shape)
    # Use a large number of terms to evaluate the kernel.
    if ndim == 1:
        kernel = kernels.ExpQuadKernel(1.2, 0.1, 3)
    elif ndim == 2:
        kernel = kernels.ExpQuadKernel(0.9, np.asarray([0.2, 0.3]), np.asarray([2.1, 2.3]))
    else:
        raise ValueError
    xs = coordgrid(*[np.linspace(0, period, n, endpoint=False) for n, period in
                     zip(shape, kernel.period * np.ones(ndim))])
    cov = kernel.evaluate(0, xs).reshape(size)
    if ndim == 1:
        rfft = np.fft.rfft(cov)
    elif ndim == 2:
        rfft = np.fft.rfft2(cov)
    else:
        raise ValueError
    assert rfft.shape == (*head, tail // 2 + 1,)
    np.testing.assert_allclose(rfft.imag, 0, atol=1e-9)

    direct_rfft = kernel.evaluate_rfft(size)
    poly = np.polynomial.Polynomial.fit(rfft.ravel(), direct_rfft.ravel(), 1).convert()
    bias, slope = poly.coef
    assert abs(bias) < 1e-2
    assert abs(slope - 1) < 1e-2


def test_matern_invalid_dof() -> None:
    with pytest.raises(ValueError):
        kernels.MaternKernel(1, 1, 1)


@pytest.mark.parametrize("dof", [3 / 2, 5 / 2])
@pytest.mark.parametrize("size", [500, (501,), (500, 502), (500, 501), (501, 500), (501, 503)])
def test_matern_approximate_rfft(dof: float, size: tuple[int]) -> None:
    sigma = 1.2
    shape = [size] if isinstance(size, int) else size
    ndim = len(shape)
    period = np.asarray([2.1, 1.7])[:ndim]
    length_scale = np.asarray([0.01, 0.02])[:ndim]
    kernel = kernels.MaternKernel(dof, sigma, length_scale)
    periodic_kernel = kernels.MaternKernel(dof, sigma, length_scale, period)
    xs = coordgrid(*(np.linspace(0, p, n, endpoint=False) for n, p in zip(shape, period)))
    xs = np.minimum(xs, period - xs)
    cov = kernel.evaluate(np.zeros(ndim), xs).reshape(shape)
    if ndim == 1:
        rfft = np.fft.rfft(cov)
    elif ndim == 2:
        rfft = np.fft.rfft2(cov)
    else:
        raise ValueError(ndim)
    np.testing.assert_allclose(rfft.imag, 0, atol=1e-9)
    rfft = rfft.real
    direct_rfft = periodic_kernel.evaluate_rfft(size)
    poly = np.polynomial.Polynomial.fit(rfft.ravel(), direct_rfft.ravel(), 1).convert()
    bias, slope = poly.coef
    assert abs(bias) < 1e-2
    assert abs(slope - 1) < 1e-2
