from gptools.stan import compile_model
import numpy as np
import pathlib
import pytest

shapes = [(3,), (4,), (3, 5), (3, 6), (4, 5), (4, 6)]


@pytest.mark.parametrize("shape", shapes, ids=["-".join(map(str, shape)) for shape in shapes])
def test_stan_numpy_fft_identity(shape: tuple[int]):
    x = np.random.normal(0, 1, shape)
    stan_file = pathlib.Path(__file__).parent / f"test_fft_identity_{x.ndim}d.stan"
    model = compile_model(stan_file=stan_file)
    data = {"x": x, "n": shape[0]}
    if x.ndim == 1:
        np_fft = np.fft.fft(x)
        np_rfft = np.fft.rfft(x)
    elif x.ndim == 2:
        data["m"] = shape[1]
        np_fft = np.fft.fft2(x)
        np_rfft = np.fft.rfft2(x)
    else:
        raise NotImplementedError
    fit = model.sample(data, fixed_param=True, iter_warmup=0, iter_sampling=1, sig_figs=9)

    # Verify the full Fourier transform.
    stan_fft, = fit.stan_variable("y")
    stan_inv_fft, = fit.stan_variable("z")
    np.testing.assert_allclose(stan_fft.real, np_fft.real, atol=1e-6)
    np.testing.assert_allclose(stan_fft.imag, np_fft.imag, atol=1e-6)
    np.testing.assert_allclose(stan_inv_fft.imag, 0, atol=1e-6)
    np.testing.assert_allclose(stan_inv_fft.real, x, atol=1e-6)

    # Verify the real Fourier transform.
    stan_rfft, = fit.stan_variable("ry")
    stan_inv_rfft, = fit.stan_variable("rz")
    np.testing.assert_allclose(stan_rfft.real, np_rfft.real, atol=1e-6)
    np.testing.assert_allclose(stan_rfft.imag, np_rfft.imag, atol=1e-6)
    np.testing.assert_allclose(stan_inv_rfft.imag, 0, atol=1e-6)
    np.testing.assert_allclose(stan_inv_rfft.real, x, atol=1e-6)
