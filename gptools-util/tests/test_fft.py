from gptools.util import coordgrid, fft, kernels
import numpy as np
import pytest
from scipy import stats
import torch as th

rfft2_shapes = [(4, 6), (4, 7), (5, 6), (5, 7)]


@pytest.fixture(params=[(), (11,), (13, 11)])
def batch_shape(request: pytest.FixtureRequest) -> tuple[int]:
    return request.param


def test_log_prob_norm(use_torch: bool) -> None:
    if use_torch:
        x = th.randn([])
    else:
        x = np.random.normal()
    loc = np.random.normal()
    scale = np.random.gamma(2)
    np.testing.assert_allclose(stats.norm(loc, scale).logpdf(x), fft.log_prob_norm(x, loc, scale))


@pytest.mark.parametrize("n", [4, 5, 9, 10])
def test_evaluate_log_prob_rfft(batch_shape: tuple[int], n: int, use_torch: bool) -> None:
    x = np.linspace(0, 1, n, endpoint=False)
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    cov = kernel(x[:, None])
    dist = stats.multivariate_normal(np.zeros_like(x), cov)
    y = dist.rvs(batch_shape)
    log_prob = dist.logpdf(y)
    log_prob_rfft = fft.evaluate_log_prob_rfft(th.as_tensor(y) if use_torch else y,
                                               th.as_tensor(cov[0]) if use_torch else cov[0])
    np.testing.assert_allclose(log_prob, log_prob_rfft)


@pytest.mark.parametrize("n", [4, 5, 9, 10])
def test_transform_rfft_roundtrip(batch_shape: tuple[int], n: int, use_torch: bool) -> None:
    x = np.linspace(0, 1, n, endpoint=False)
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    cov = kernel(x[:, None])[0]
    z = np.random.normal(0, 1, n)
    z = th.as_tensor(z) if use_torch else z
    cov = th.as_tensor(cov) if use_torch else cov
    y = fft.transform_irfft(z, cov)
    x = fft.transform_rfft(y, cov)
    # Verify that the inverse of the transform is the input.
    np.testing.assert_allclose(z, x)


@pytest.mark.parametrize("shape", rfft2_shapes,
                         ids=["-".join(map(str, shape)) for shape in rfft2_shapes])
def test_evaluate_log_prob_rfft2(batch_shape: tuple[int], shape: int, use_torch: bool) -> None:
    xs = coordgrid(*(np.linspace(0, 1, size, endpoint=False) for size in shape))
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    cov = kernel(xs)
    dist = stats.multivariate_normal(np.zeros(np.prod(shape)), cov)
    y = dist.rvs(batch_shape)
    log_prob = dist.logpdf(y)
    y2 = y.reshape(batch_shape + shape)
    cov2 = cov[0].reshape(shape)
    if use_torch:
        y2 = th.as_tensor(y2)
        cov2 = th.as_tensor(cov2)
    log_prob_rfft2 = fft.evaluate_log_prob_rfft2(y2, cov2)
    np.testing.assert_allclose(log_prob, log_prob_rfft2)
