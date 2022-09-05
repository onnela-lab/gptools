import cmdstanpy
from gptools.util.kernels import ExpQuadKernel
from gptools.stan import get_include
import numpy as np
import pathlib
import pytest
from scipy import stats


@pytest.fixture(params=[20, 21], scope="session")
def data_1d(request: pytest.FixtureRequest) -> dict:
    n: int = request.param
    x = np.arange(n)
    ys = []
    covs = []
    log_probs = []
    m = 100
    for _ in range(m):
        kernel = ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.1), 1e-2, x.size)
        cov = kernel(x[:, None])
        dist = stats.multivariate_normal(np.zeros_like(x), cov)
        y = dist.rvs()
        ys.append(y)
        covs.append(cov)
        log_probs.append(dist.logpdf(y))

    covs = np.asarray(covs)
    log_probs = np.asarray(log_probs)
    ys = np.asarray(ys)

    assert covs.shape == (m, n, n)
    assert log_probs.shape == (m,)
    assert ys.shape == (m, n)

    return {
        "m": m,
        "n": n,
        "x": x,
        "ys": ys,
        "kernels": kernel,
        "covs": covs,
        "log_probs": log_probs,
    }


@pytest.fixture(scope="session")
def fft_gp_model() -> cmdstanpy.CmdStanModel:
    stan_file = pathlib.Path(__file__).parent / "test_fft_gp.stan"
    return cmdstanpy.CmdStanModel(stan_file=stan_file, compile="force",
                                  stanc_options={"include-paths": [get_include()]})


def test_log_prob_fft_1d(data_1d: dict, fft_gp_model: cmdstanpy.CmdStanModel) -> None:
    data = data_1d
    m = data["m"]
    n = data["n"]
    # Evaluate the fft of the kernel and samples.
    fftvar = n * np.fft.rfft(data["covs"][:, 0])
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = np.fft.rfft(data["ys"])

    # Check shapes.
    shape = (m, n // 2 + 1)
    assert fftvar.shape == shape
    assert ffts.shape == shape

    # Ensure imaginary parts are zero where expected.
    np.testing.assert_allclose(ffts[:, 0].imag, 0, atol=1e-9)
    if n % 2 == 0:
        np.testing.assert_allclose(ffts[:, n // 2].imag, 0, atol=1e-9)

    # Evaluate the scales and rescale the Fourier coefficients.
    fft_scale = np.sqrt(fftvar / 2)
    fft_scale[:, 0] *= np.sqrt(2)

    if n % 2 == 0:
        fft_scale[:, n // 2] *= np.sqrt(2)

    # Scale the fourier transforms and evaluate the likelihood.
    iweight = np.ones(n // 2 + 1)
    iweight[0] = 0
    if n % 2 == 0:
        iweight[n // 2] = 0

    log_prob = stats.norm(0, fft_scale).logpdf(ffts.real).sum(axis=-1) \
        + stats.norm(0, fft_scale).logpdf(ffts.imag) @ iweight \
        - np.log(2) * ((n - 1) // 2) + n * np.log(n) / 2
    np.testing.assert_allclose(log_prob, data["log_probs"])

    # Compare with the Stan implementation.
    for i, (y, cov) in enumerate(zip(data["ys"], data["covs"])):
        fit = fft_gp_model.sample({"n": n, "y": y, "cov": cov[0]}, iter_sampling=1, iter_warmup=0,
                                  fixed_param=True, sig_figs=9)
        np.testing.assert_allclose(log_prob[i], fit.stan_variable("log_prob")[0])


@pytest.mark.parametrize("n", [3, 4])
def test_stan_numpy_fft_equivalence(n: int):
    stan_file = pathlib.Path(__file__).parent / "test_fft.stan"
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    x = np.random.normal(0, 1, n)
    fit = model.sample({"n": n, "x": x}, fixed_param=True, iter_warmup=0, iter_sampling=1,
                       sig_figs=9)
    stan_fft, = fit.stan_variable("y")
    np_fft = np.fft.fft(x)
    np.testing.assert_allclose(stan_fft.real, np_fft.real)
    np.testing.assert_allclose(stan_fft.imag, np_fft.imag)
