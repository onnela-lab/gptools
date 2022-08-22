from gptools.kernels import ExpQuadKernel
import numpy as np
import pytest
from scipy import stats


@pytest.fixture(params=[20, 21])
def data(request: pytest.FixtureRequest) -> dict:
    n: int = request.param
    x = np.arange(n)
    ys = []
    covs = []
    log_probs = []
    m = 17
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


@pytest.mark.parametrize("method", ["fft", "rfft"])
@pytest.mark.parametrize("dist", ["norm", "chi2"])
def test_log_prob_fft_normal(data: dict, method: str, dist: str) -> None:
    m = data["m"]
    n = data["n"]
    fft = getattr(np.fft, method)
    # Evaluate the fft of the kernel and samples.
    fftvar = n * fft(data["covs"][:, 0])
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = fft(data["ys"])

    # Check shapes.
    if method == "rfft":
        shape = (m, n // 2 + 1)
    else:
        shape = (m, n)
    assert fftvar.shape == shape
    assert ffts.shape == shape

    # Get the scales.
    fft_scale = np.sqrt(fftvar / 2)
    fft_scale[:, 0] *= np.sqrt(2)

    scaled_ffts = ffts / fft_scale
    if n % 2 == 0:
        scaled_ffts[:, n // 2] /= np.sqrt(2)

    # Scale the fourier transforms and evaluate the likelihood.
    if dist == "norm":
        rweight = np.ones(n // 2 + 1 if method == "rfft" else n)
        iweight = np.ones_like(rweight)
        rweight[n // 2 + 1:] = 0
        iweight[n // 2 + 1:] = 0
        iweight[0] = 0
        if n % 2 == 0:
            iweight[n // 2] = 0
        log_prob = stats.norm().logpdf(scaled_ffts.real) @ rweight \
            + stats.norm().logpdf(scaled_ffts.imag) @ iweight \
            - np.log(fft_scale) @ (iweight + rweight) \
            + np.log(2) / 2 * (1 - n) + n * np.log(n) / 2
    elif dist == "chi2":
        pytest.skip()
    else:
        raise ValueError(dist)
    np.testing.assert_allclose(log_prob, data["log_probs"])
