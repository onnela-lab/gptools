from gptools.torch.fft import FourierGaussianProcess1D, FourierGaussianProcess1DTransform
from gptools.util.kernels import ExpQuadKernel
from gptools.util import coordgrid
import numpy as np
import pytest
from scipy import stats
import torch as th


shapes = [(20,), (21,), (7, 9), (8, 9), (7, 8), (8, 6)]


@pytest.fixture(params=shapes, ids=["-".join(map(str, shape)) for shape in shapes])
def shape(request: pytest.FixtureRequest) -> tuple[int]:
    return request.param


@pytest.fixture
def data(shape: tuple[int]) -> dict:
    if len(shape) > 1:
        pytest.skip("not yet implemented for higher dimensions")
    xs = coordgrid(*(np.arange(size) for size in shape))
    size = np.prod(shape)
    assert xs.shape == (size, len(shape))
    kernel = ExpQuadKernel(np.random.gamma(10, 0.01), np.random.gamma(10, 0.1), 0.1, shape)
    cov = th.as_tensor(kernel(xs))
    loc = th.randn(size)
    dist = stats.multivariate_normal(loc, cov)
    y = th.as_tensor(dist.rvs())

    return {
        "ndim": len(shape),
        "shape": shape,
        "xs": xs,
        "y": y.reshape(shape),
        "kernels": kernel,
        "cov": cov[0].reshape(shape),
        "log_prob": dist.logpdf(y),
        "loc": loc,
        "size": size,
    }


def test_log_prob_fft(data: dict) -> None:
    dist = FourierGaussianProcess1D(data["loc"], data["cov"])
    np.testing.assert_allclose(dist.log_prob(data["y"]), data["log_prob"])


def test_fft_gp_transform_roundtrip(data: dict) -> None:
    z = th.randn(data["size"])
    transform = FourierGaussianProcess1DTransform(th.zeros_like(data["loc"]), data["cov"])
    y = transform(z)
    x = transform.inv(y)
    np.testing.assert_allclose(z, x)
