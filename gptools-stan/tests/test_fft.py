from gptools.util.kernels import ExpQuadKernel
from gptools.util import coordgrid
from gptools.stan import compile_model
import numpy as np
import pathlib
import pytest
from scipy import stats

shapes = [(20,), (21,), (7, 9), (8, 9), (7, 8), (8, 6)]


@pytest.fixture(params=shapes, ids=["-".join(map(str, shape)) for shape in shapes], scope="session")
def data(request: pytest.FixtureRequest) -> dict:
    shape: tuple[int] = request.param
    xs = coordgrid(*(np.arange(size) for size in shape))
    size = np.prod(shape)
    assert xs.shape == (size, len(shape))
    kernel = ExpQuadKernel(np.random.gamma(10, 0.01), np.random.gamma(10, 0.1), 0.1, shape)
    cov = kernel(xs)
    dist = stats.multivariate_normal(np.zeros(xs.shape[0]), cov)
    y = dist.rvs()

    return {
        "ndim": len(shape),
        "shape": shape,
        "xs": xs,
        "y": y.reshape(shape),
        "kernels": kernel,
        "cov": cov[0].reshape(shape),
        "log_prob": dist.logpdf(y),
    }


def test_log_prob_fft(data: dict) -> None:
    stan_file = pathlib.Path(__file__).parent / f"test_fft_gp_{data['ndim']}d.stan"
    fft_gp_model = compile_model(stan_file=stan_file)
    stan_data = {"n": data["shape"][0], "y": data["y"], "cov": data["cov"]}
    if data["ndim"] == 2:
        stan_data["m"] = data["shape"][1]
    fit = fft_gp_model.sample(stan_data, iter_sampling=1, iter_warmup=0, fixed_param=True,
                              sig_figs=9)
    np.testing.assert_allclose(data["log_prob"], fit.stan_variable("log_prob")[0])


@pytest.mark.parametrize("shape", [(3,), (4,), (3, 5), (3, 6), (4, 5), (4, 6)])
def test_stan_numpy_fft_identity(shape: tuple[int]):
    x = np.random.normal(0, 1, shape)
    stan_file = pathlib.Path(__file__).parent / f"test_fft_identity_{x.ndim}d.stan"
    model = compile_model(stan_file=stan_file)
    data = {"x": x, "n": shape[0]}
    if x.ndim == 1:
        np_fft = np.fft.fft(x)
    elif x.ndim == 2:
        data["m"] = shape[1]
        np_fft = np.fft.fft2(x)
    else:
        raise NotImplementedError
    fit = model.sample(data, fixed_param=True, iter_warmup=0, iter_sampling=1, sig_figs=9)
    stan_fft, = fit.stan_variable("y")
    stan_inv_fft, = fit.stan_variable("z")
    np.testing.assert_allclose(stan_fft.real, np_fft.real, atol=1e-6)
    np.testing.assert_allclose(stan_fft.imag, np_fft.imag, atol=1e-6)
    np.testing.assert_allclose(stan_inv_fft.imag, 0, atol=1e-6)
    np.testing.assert_allclose(stan_inv_fft.real, x, atol=1e-6)
