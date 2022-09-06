import cmdstanpy
from gptools.util.kernels import ExpQuadKernel
from gptools.util import coordgrid
from gptools.stan import get_include
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
    ys = []
    covs = []
    log_probs = []
    num_samples = 3
    for _ in range(num_samples):
        kernel = ExpQuadKernel(np.random.gamma(10, 0.01), np.random.gamma(10, 0.1), 0.1, shape)
        cov = kernel(xs)
        dist = stats.multivariate_normal(np.zeros(xs.shape[0]), cov)
        y = dist.rvs()
        ys.append(y)
        covs.append(cov)
        log_probs.append(dist.logpdf(y))

    covs = np.asarray(covs)
    log_probs = np.asarray(log_probs)
    ys = np.asarray(ys)

    assert covs.shape == (num_samples, size, size)
    assert log_probs.shape == (num_samples,)
    assert ys.shape == (num_samples, size)

    return {
        "ndim": len(shape),
        "num_samples": num_samples,
        "shape": shape,
        "xs": xs,
        "ys": ys,
        "kernels": kernel,
        "covs": covs,
        "log_probs": log_probs,
    }


@pytest.fixture(scope="session")
def fft_gp_model(data: dict) -> cmdstanpy.CmdStanModel:
    stan_file = pathlib.Path(__file__).parent / f"test_fft_gp_{data['ndim']}d.stan"
    return cmdstanpy.CmdStanModel(stan_file=stan_file,
                                  stanc_options={"include-paths": [get_include()]})


def test_log_prob_fft(data: dict, fft_gp_model: cmdstanpy.CmdStanModel) -> None:
    if data["ndim"] == 1:
        _test_log_prob_fft_1d(data, fft_gp_model)
    elif data["ndim"] == 2:
        _test_log_prob_fft_2d(data, fft_gp_model)
    else:
        raise ValueError


def _test_log_prob_fft_1d(data: dict, fft_gp_model: cmdstanpy.CmdStanModel) -> None:
    num_samples = data["num_samples"]
    size, = data["shape"]
    # Evaluate the fft of the kernel and samples.
    fftvar = size * np.fft.rfft(data["covs"][:, 0])
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = np.fft.rfft(data["ys"])

    # Check shapes.
    shape = (num_samples, size // 2 + 1)
    assert fftvar.shape == shape
    assert ffts.shape == shape

    # Ensure imaginary parts are zero where expected.
    np.testing.assert_allclose(ffts[:, 0].imag, 0, atol=1e-9)
    if size % 2 == 0:
        np.testing.assert_allclose(ffts[:, size // 2].imag, 0, atol=1e-9)

    # Evaluate the scales of the Fourier coefficients.
    fft_scale = np.sqrt(fftvar / 2)
    fft_scale[:, 0] *= np.sqrt(2)

    if size % 2 == 0:
        fft_scale[:, size // 2] *= np.sqrt(2)

    # Scale the fourier transforms and evaluate the likelihood.
    iweight = np.ones(size // 2 + 1)
    iweight[0] = 0
    if size % 2 == 0:
        iweight[size // 2] = 0

    log_prob = stats.norm(0, fft_scale).logpdf(ffts.real).sum(axis=-1) \
        + stats.norm(0, fft_scale).logpdf(ffts.imag) @ iweight \
        - np.log(2) * ((size - 1) // 2) + size * np.log(size) / 2
    np.testing.assert_allclose(log_prob, data["log_probs"])

    # Compare with the Stan implementation.
    for i, (y, cov) in enumerate(zip(data["ys"], data["covs"])):
        fit = fft_gp_model.sample({"n": size, "y": y, "cov": cov[0]}, iter_sampling=1,
                                  iter_warmup=0, fixed_param=True, sig_figs=9)
        np.testing.assert_allclose(log_prob[i], fit.stan_variable("log_prob")[0])


def _test_log_prob_fft_2d(data: dict, fft_gp_model: cmdstanpy.CmdStanModel) -> None:
    num_samples = data["num_samples"]
    height, width = shape = data["shape"]

    # Reshape the samples and covariances to their "natural" shape.
    ys: np.ndarray = data["ys"].reshape((num_samples, *shape))
    covs: np.ndarray = data["covs"][:, 0].reshape((num_samples, *shape))

    # Evaluate the fft of the kernel and samples.
    fftvar = np.prod(shape) * np.fft.rfft2(covs)
    np.testing.assert_allclose(fftvar.imag, 0, atol=1e-9)
    fftvar = fftvar.real
    ffts = np.fft.rfft2(ys)

    # Check shapes.
    fftshape = (num_samples, height, width // 2 + 1)
    assert fftvar.shape == fftshape
    assert ffts.shape == fftshape

    # Evaluate the scales of Fourier coefficients. The division by two accounts for half of the
    # variance going to real and imaginary terms each. We subsequently make adjustments based on
    # which elements are purely real. We also construct a binary mask for which elements should be
    # included in the likelihood evaluation.
    fftscale = np.sqrt(fftvar / 2)
    imask = np.ones(fftshape[1:])
    rmask = np.ones(fftshape[1:])

    # Recall how the two-dimensional RFFT is computed. We first take an RFFT of rows of the matrix.
    # This leaves us with a real first column (zero frequency term) and a real last column if the
    # number of columns is even (Nyqvist frequency term). Second, we take a *full* FFT of the
    # columns. The first column will have a real coefficient in the first row (zero frequency in the
    # "row-dimension"). All elements in rows beyond n // 2 + 1 are irrelevant because the column was
    # real. The same applies to the last column if there is a Nyqvist frequency term. Finally, we
    # will also have a real-only Nyqvist frequency term in the first and last column if the number
    # of rows is even.

    # Sanity check of imaginary parts that should be zero. The first is the zero-frequency term in
    # both dimensions which must always be real.
    np.testing.assert_allclose(ffts[:, 0, 0].imag, 0, atol=1e-9)
    fftscale[:, 0, 0] *= np.sqrt(2)
    imask[0, 0] = 0
    # We mask out the last elements of the first column because they are redundant (because the
    # first column is real after the column FFT).
    imask[height // 2 + 1:, 0] = 0
    rmask[height // 2 + 1:, 0] = 0

    # If the width is even, we get a real last column after the first transform due to the Nyqvist
    # frequency.
    if width % 2 == 0:
        np.testing.assert_allclose(ffts[:, 0, -1].imag, 0, atol=1e-9)
        fftscale[:, 0, -1] *= np.sqrt(2)
        imask[0, -1] = 0
        imask[height // 2 + 1:, -1] = 0
        rmask[height // 2 + 1:, -1] = 0

    # If the height is even, we get an extra Nyqvist frequency term in the first column.
    if height % 2 == 0:
        np.testing.assert_allclose(ffts[:, height // 2, 0].imag, 0, atol=1e-9)
        fftscale[:, height // 2, 0] *= np.sqrt(2)
        imask[height // 2, 0] = 0

    # If the height and width are even, the Nyqvist frequencies in the first and last column must
    # be real.
    if width % 2 == 0 and height % 2 == 0:
        np.testing.assert_allclose(ffts[:, height // 2, -1].imag, 0, atol=1e-9)
        fftscale[:, height // 2, -1] *= np.sqrt(2)
        imask[height // 2, -1] = 0

    # At this point, all real and imaginary components of the Fourier coefficients should have the
    # correct scale (except the imaginary part where it is missing). The masks tell us which
    # coefficients to consider in the likelihood--which we now evaluate. But first, let's check that
    # the number of independent parameters matches up (there can't be more or less information in
    # Fourier space).
    assert imask.sum() + rmask.sum() == np.prod(shape)

    size = np.prod(shape)
    nterms = (size - 1) // 2
    if height % 2 == 0 and width % 2 == 0:
        nterms -= 1
    log_prob = (stats.norm(0, fftscale).logpdf(ffts.real) * rmask).sum(axis=(-1, -2)) + \
        (stats.norm(0, fftscale).logpdf(ffts.imag) * imask).sum(axis=(-1, -2)) \
        - np.log(2) * nterms + size * np.log(size) / 2
    np.testing.assert_allclose(log_prob, data["log_probs"])

    # Compare with the Stan implementation.
    log_prob_stan = []
    for y, cov in zip(ys, covs):
        fit = fft_gp_model.sample({"n": height, "m": width, "y": y, "cov": cov}, iter_sampling=1,
                                  iter_warmup=0, fixed_param=True, sig_figs=9)
        log_prob_stan.append(fit.stan_variable("log_prob")[0])
    np.testing.assert_allclose(log_prob, log_prob_stan)


@pytest.mark.parametrize("shape", [(3,), (4,), (3, 5), (3, 6), (4, 5), (4, 6)])
def test_stan_numpy_fft_identity(shape: tuple[int]):
    x = np.random.normal(0, 1, shape)
    stan_file = pathlib.Path(__file__).parent / f"test_fft_identity_{x.ndim}d.stan"
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
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
    np.testing.assert_allclose(stan_fft.real, np_fft.real, atol=1e-6)
    np.testing.assert_allclose(stan_fft.imag, np_fft.imag, atol=1e-6)
