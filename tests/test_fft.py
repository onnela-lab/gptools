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


@pytest.mark.parametrize("rfft", [False, True])
def test_log_prob_fft_normal(data: dict, rfft: bool) -> None:
    fft = np.fft.rfft if rfft else np.fft.fft
    # Evaluate the fft of the kernel and samples.
    fftvar = fft(data["cov"][0])
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = fft(data["ys"])

    # Check shapes.
    n = data["n"]
    if rfft:
        assert fftvar.shape == (n // 2 + 1,)
    else:
        assert fftvar.shape == (n,)

    # Scale the fourier transforms and evaluate the likelihood.
    scaled_ffts = ffts / np.sqrt(fftvar)
    scaled_ffts[:, 1:] *= np.sqrt(2)
    rweight = np.ones_like(fftvar)
    iweight = np.ones_like(fftvar)
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
