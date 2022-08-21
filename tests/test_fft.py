from gptools.kernels import ExpQuadKernel
import numpy as np
import pytest
from scipy import stats


@pytest.fixture(params=[20, 21])
def data(request: pytest.FixtureRequest) -> dict:
    x = np.arange(request.param)
    kernel = ExpQuadKernel(1, 3, 1e-2, x.size)
    cov = kernel(x[:, None])
    dist = stats.multivariate_normal(np.zeros_like(x), cov)
    ys = dist.rvs(100)
    return {
        "n": x.size,
        "x": x,
        "ys": ys,
        "kernel": kernel,
        "dist": dist,
        "cov": cov,
        "log_prob": dist.logpdf(ys),
    }


def test_log_prob_fft_normal(data: dict) -> None:
    # Evaluate the fft of the kernel and samples.
    fftvar = np.fft.fft(data["cov"][0])
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = np.fft.fft(data["ys"])

    # Scale the fourier transforms and evaluate the likelihood.
    scaled_ffts = ffts / np.sqrt(fftvar)
    scaled_ffts[:, 1:] *= np.sqrt(2)
    n = data["n"]
    rweight = np.ones(n)
    iweight = np.ones(n)
    if n % 2:
        iweight[0] = 0
        rweight[n // 2 + 1:] = 0
        iweight[n // 2 + 1:] = 0
    else:
        iweight[0] = 0
        rweight[n // 2] = 0.5
        iweight[n // 2] = 0.5
        rweight[n // 2 + 1:] = 0
        iweight[n // 2 + 1:] = 0
    log_prob = stats.norm().logpdf(scaled_ffts.real) @ rweight \
        + stats.norm().logpdf(scaled_ffts.imag) @ iweight
    # We don't test for equality because we don't evaluate the Jacobian of the Fourier transform.
    pearsonr, _ = stats.pearsonr(log_prob, data["log_prob"])
    np.testing.assert_allclose(pearsonr, 1)


def test_log_prob_rfft_normal(data: dict) -> None:
    # Evaluate the fft of the kernel and samples.
    fftvar = np.fft.rfft(data["cov"][0])
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = np.fft.rfft(data["ys"])

    # Scale the fourier transforms and evaluate the likelihood.
    scaled_ffts = ffts / np.sqrt(fftvar)
    scaled_ffts[:, 1:] *= np.sqrt(2)
    n = data["n"]
    rweight = np.ones(n // 2 + 1)
    iweight = np.ones(n // 2 + 1)
    if n % 2:
        iweight[0] = 0
    else:
        iweight[0] = 0
        rweight[n // 2] = 0.5
        iweight[n // 2] = 0.5
    log_prob = stats.norm().logpdf(scaled_ffts.real) @ rweight \
        + stats.norm().logpdf(scaled_ffts.imag) @ iweight
    # We don't test for equality because we don't evaluate the Jacobian of the Fourier transform.
    pearsonr, _ = stats.pearsonr(log_prob, data["log_prob"])
    np.testing.assert_allclose(pearsonr, 1)
