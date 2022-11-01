from gptools.torch.fft import FourierGaussianProcess1D, FourierGaussianProcess1DTransform, \
    FourierGaussianProcess2D, FourierGaussianProcess2DTransform
from gptools.util.kernels import DiagonalKernel, ExpQuadKernel
from gptools.util import coordgrid
import numpy as np
import pytest
from scipy import stats
import torch as th
from typing import Optional


shapes = [(20,), (21,), (7, 9), (8, 9), (7, 8), (8, 6)]


@pytest.fixture(params=shapes, ids=["-".join(map(str, shape)) for shape in shapes])
def shape(request: pytest.FixtureRequest) -> tuple[int]:
    return request.param


@pytest.fixture
def data(shape: tuple[int]) -> dict:
    ndim = len(shape)
    xs = coordgrid(*(np.arange(size) for size in shape))
    size = np.prod(shape)
    assert xs.shape == (size, ndim)
    kernel = ExpQuadKernel(np.random.gamma(10, 0.01), np.random.gamma(10, 0.1), np.asarray(shape)) \
        + DiagonalKernel(0.1, shape)
    cov = th.as_tensor(kernel.evaluate(xs))
    loc = th.randn(shape)
    dist = stats.multivariate_normal(loc.ravel(), cov)
    y = th.as_tensor(dist.rvs())

    result = {
        "ndim": ndim,
        "shape": shape,
        "xs": xs,
        "y": y.reshape(shape),
        "kernels": kernel,
        "cov": cov[0].reshape(shape),
        "log_prob": dist.logpdf(y),
        "loc": loc,
        "size": size,
    }

    if ndim == 1:
        result.update({
            "distribution_cls": FourierGaussianProcess1D,
            "transform_cls": FourierGaussianProcess1DTransform,
        })
    elif ndim == 2:
        result.update({
            "distribution_cls": FourierGaussianProcess2D,
            "transform_cls": FourierGaussianProcess2DTransform,
        })
    else:
        raise ValueError(ndim)

    return result


def test_log_prob_fft(data: dict) -> None:
    dist = data["distribution_cls"](data["loc"], cov=data["cov"])
    np.testing.assert_allclose(dist.log_prob(data["y"]), data["log_prob"])


def test_fft_gp_transform_roundtrip(data: dict) -> None:
    z = th.randn(data["shape"])
    transform = data["transform_cls"](data["loc"], cov=data["cov"])
    y = transform(z)
    x = transform.inv(y)
    np.testing.assert_allclose(z, x)


@pytest.mark.parametrize("sample_shape", [(), (2,), (3, 4)])
def test_fft_gp_sample(data: dict, sample_shape: Optional[th.Size]) -> None:
    dist = data["distribution_cls"](data["loc"], cov=data["cov"])
    assert dist.has_rsample
    assert dist.rsample(sample_shape).shape == sample_shape + data["shape"]
