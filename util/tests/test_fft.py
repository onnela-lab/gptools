from gptools.util import ArrayOrTensorDispatch, coordgrid, fft, kernels
import numpy as np
import pytest
from scipy import stats
import torch as th

rfft2_shapes = [(4, 6), (4, 7), (5, 6), (5, 7)]


@pytest.fixture(params=rfft2_shapes, ids=["-".join(map(str, shape)) for shape in rfft2_shapes])
def rfft2_shape(request: pytest.FixtureRequest) -> tuple[int]:
    return request.param


@pytest.fixture(params=[4, 5, 9, 10])
def rfft_num(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(params=[(), (11,), (13, 11)])
def batch_shape(request: pytest.FixtureRequest) -> tuple[int]:
    return request.param


def test_log_prob_norm(use_torch: bool) -> None:
    x = th.randn([]) if use_torch else np.random.normal()
    np.testing.assert_allclose(stats.norm(0, 1).logpdf(x), fft.log_prob_stdnorm(x))


def test_evaluate_log_prob_rfft(batch_shape: tuple[int], rfft_num: int, use_torch: bool) -> None:
    x = np.linspace(0, 1, rfft_num, endpoint=False)
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1) \
        + kernels.DiagonalKernel(0.1, 1)
    cov = kernel.evaluate(x[:, None])
    loc = np.random.normal(0, 1, rfft_num)
    dist = stats.multivariate_normal(loc, cov)
    y = dist.rvs(batch_shape)
    log_prob = dist.logpdf(y)
    lincov = cov[0]
    if use_torch:
        y = th.as_tensor(y)
        loc = th.as_tensor(loc)
        lincov = th.as_tensor(lincov)
    log_prob_rfft = fft.evaluate_log_prob_rfft(y, loc, cov=lincov)
    np.testing.assert_allclose(log_prob, log_prob_rfft)
    cov_rfft = ArrayOrTensorDispatch()[cov].fft.rfft(lincov)
    log_prob_rfft = fft.evaluate_log_prob_rfft(y, loc, cov_rfft=cov_rfft)
    np.testing.assert_allclose(log_prob, log_prob_rfft)


def test_transform_rfft_roundtrip(batch_shape: tuple[int], rfft_num: int, use_torch: bool) -> None:
    x = np.linspace(0, 1, rfft_num, endpoint=False)
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1) \
        + kernels.DiagonalKernel(0.1, 1)
    z = np.random.normal(0, 1, (*batch_shape, rfft_num))
    loc = np.random.normal(0, 1, rfft_num)
    cov = kernel.evaluate(x[:, None])
    cov = cov[0]
    if use_torch:
        loc = th.as_tensor(loc)
        z = th.as_tensor(z)
        cov = th.as_tensor(cov)
    y = fft.transform_irfft(z, loc, cov=cov)
    x = fft.transform_rfft(y, loc, cov=cov)
    # Verify that the inverse of the transform is the input.
    np.testing.assert_allclose(z, x)


@pytest.mark.parametrize("kernel", [
    kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1.2),
    kernels.MaternKernel(1.5, np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1.3),
    kernels.MaternKernel(2.5, np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1.4),
])
def test_evaluate_log_prob_rfft2(kernel: kernels.Kernel, batch_shape: tuple[int], rfft2_shape: int,
                                 use_torch: bool) -> None:
    xs = coordgrid(*(np.linspace(0, kernel.period, size, endpoint=False) for size in rfft2_shape))
    kernel = kernel + kernels.DiagonalKernel(1e-2, kernel.period)
    cov = kernel.evaluate(xs)
    loc = np.random.normal(0, 1, xs.shape[0])
    dist = stats.multivariate_normal(loc, cov)
    y = dist.rvs(batch_shape)
    log_prob = dist.logpdf(y)
    y2 = y.reshape(batch_shape + rfft2_shape)
    loc2 = loc.reshape(rfft2_shape)
    cov2 = cov[0].reshape(rfft2_shape)
    if use_torch:
        y2 = th.as_tensor(y2)
        loc2 = th.as_tensor(loc2)
        cov2 = th.as_tensor(cov2)
    log_prob_rfft2 = fft.evaluate_log_prob_rfft2(y2, loc2, cov=cov2)
    np.testing.assert_allclose(log_prob, log_prob_rfft2)
    cov_rfft2 = ArrayOrTensorDispatch()[cov].fft.rfft2(cov2)
    log_prob_rfft = fft.evaluate_log_prob_rfft2(y2, loc2, cov_rfft2=cov_rfft2)
    np.testing.assert_allclose(log_prob, log_prob_rfft)


def test_pack_rfft2_roundtrip(batch_shape: tuple[int], rfft2_shape: int, use_torch: bool) -> None:
    z = np.random.normal(0, 1, batch_shape + rfft2_shape)
    if use_torch:
        z = th.as_tensor(z)

    # Unpacked Fourier to Fourier and back.
    y = fft.fft2.pack_rfft2(z)
    x = fft.fft2.unpack_rfft2(y, z.shape)
    np.testing.assert_allclose(z, x)

    # Fourier to unpacked Fourier and back.
    dispatch = ArrayOrTensorDispatch()
    z = dispatch[z].fft.rfft2(z)
    y = fft.fft2.unpack_rfft2(z, rfft2_shape)
    x = fft.fft2.pack_rfft2(y)
    np.testing.assert_allclose(z, x)


def test_transform_rfft2_roundtrip(batch_shape: tuple[int], rfft2_shape: int, use_torch: bool) \
        -> None:
    xs = coordgrid(*(np.linspace(0, 1, size, endpoint=False) for size in rfft2_shape))
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1) \
        + kernels.DiagonalKernel(1e-3, 1)
    cov = kernel.evaluate(xs)
    cov = cov[0].reshape(rfft2_shape)
    loc = np.random.normal(0, 1, rfft2_shape)
    z = np.random.normal(0, 1, batch_shape + rfft2_shape)
    if use_torch:
        z = th.as_tensor(z)
        cov = th.as_tensor(cov)
    y = fft.transform_irfft2(z, loc, cov=cov)
    x = fft.transform_rfft2(y, loc, cov=cov)
    # Verify that the inverse of the transform is the input.
    np.testing.assert_allclose(z, x)


@pytest.mark.parametrize("n", [5, 7])
def test_expand_rfft(n: int) -> None:
    x = np.random.normal(0, 1, n)
    rfft = np.fft.rfft(x)
    np.testing.assert_allclose(np.fft.fft(x), fft.expand_rfft(rfft, n))
