from gptools.stan import compile_model
from gptools.util import coordgrid, fft, kernels
import hashlib
import numpy as np
import pathlib
import pytest
from scipy import stats
import typing


def assert_stan_python_allclose(
        stan_function: str, python_function: typing.Callable, arg_types: dict[str, str],
        arg_values: dict[str, np.ndarray], result_type: str, includes: typing.Iterable[str],
        desired: typing.Optional[np.ndarray] = None, atol: float = 1e-8) -> None:
    """
    Assert that a Stan and Python function return the same result up to numerical inaccuracies.
    """
    # Assemble the stan code we seek to build.
    functions = "\n".join(f"#include {include}" for include in includes)
    data = "\n".join(f"{type} {name};" for name, type in arg_types.items())
    args = [arg for arg in arg_values if not arg.endswith("_")]
    if stan_function.endswith("_lpdf"):
        x, *args = args
        generated_quantities = f"{result_type} result = {stan_function}({x} | {', '.join(args)});"
    else:
        generated_quantities = f"{result_type} result = {stan_function}({', '.join(args)});"
    code = "\n".join([
        "functions {", functions, "}",
        "data {", data, "}",
        "generated quantities {", generated_quantities, "}",
    ])

    # Write to file if it does not already exist.
    digest = hashlib.sha256(code.encode()).hexdigest()
    path = pathlib.Path(".cmdstanpy_cache", digest).with_suffix(".stan")
    if not path.is_file():
        path.parent.mkdir(exist_ok=True)
        path.write_text(code)

    # Compile the model and obtain the result.
    try:
        model = compile_model(stan_file=path)
        fit = model.sample(arg_values, fixed_param=True, iter_sampling=1, iter_warmup=1, sig_figs=9)
        result_stan, = fit.stan_variable("result")
    except Exception as ex:
        raise RuntimeError(f"failed to get Stan result for {stan_function}") from ex

    # Get the python result and compare.
    try:
        result_python = python_function(**{key: value for key, value in arg_values.items() if not
                                        key.endswith("_")})
    except Exception as ex:
        raise RuntimeError(f"failed to get Python result for {python_function}") from ex

    try:
        np.testing.assert_allclose(result_stan, result_python, atol=atol)
    except Exception as ex:
        raise RuntimeError(f"Stan ({stan_function}) and Python ({python_function}) results do not "
                           "match") from ex

    # Verify against expected value. We only check one because we have already verified that they
    # are the same.
    if desired is not None:
        try:
            np.testing.assert_allclose(result_python, desired, atol=atol)
        except Exception as ex:
            raise RuntimeError(f"results do not match desired value for {python_function}") from ex


configs = []

for n in [7, 8]:
    # One-dimensional real Fourier transform ...
    y = np.random.normal(0, 1, n)
    configs.append({
        "stan_function": "rfft",
        "python_function": np.fft.rfft,
        "arg_types": {"n_": "int", "a": "vector[n_]"},
        "arg_values": {"n_": n, "a": y},
        "result_type": "complex_vector[n_ %/% 2 + 1]",
        "includes": ["gptools_util.stan"],
    })

    # ... and its inverse.
    z = np.fft.rfft(y)
    configs.append({
        "stan_function": "inv_rfft",
        "python_function": np.fft.irfft,
        "arg_types": {"n": "int", "a": "complex_vector[n %/% 2 + 1]"},
        "arg_values": {"a": z, "n": n},
        "result_type": "vector[n]",
        "includes": ["gptools_util.stan"],
        "desired": y,
    })

    # Unpack truncated Fourier coefficients to a real vector ...
    configs.append({
        "stan_function": "gp_unpack_rfft",
        "python_function": fft.unpack_rfft,
        "arg_types": {"size": "int", "z": "complex_vector[size %/% 2 + 1]"},
        "arg_values": {"z": z, "size": n},
        "result_type": "vector[size]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
    })

    # ... and pack them up again.
    unpacked_z = fft.unpack_rfft(z, n)
    configs.append({
        "stan_function": "gp_pack_rfft",
        "python_function": fft.pack_rfft,
        "arg_types": {"n_": "int", "z": "vector[n_]"},
        "arg_values": {"n_": n, "z": unpacked_z},
        "result_type": "complex_vector[n_ %/% 2 + 1]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": z,
    })

    # Transforming to whitened Fourier coefficients ...
    loc = np.random.normal(0, 1, n)
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    cov = kernel(np.arange(n)[:, None])
    lincov = cov[0]
    configs.append({
        "stan_function": "gp_transform_rfft",
        "python_function": fft.transform_rfft,
        "arg_types": {"n_": "int", "y": "vector[n_]", "loc": "vector[n_]", "cov": "vector[n_]"},
        "arg_values": {"n_": n, "y": y, "loc": loc, "cov": lincov},
        "result_type": "vector[n_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
    })

    # ... and back again.
    z = fft.transform_rfft(y, loc, lincov)
    configs.append({
        "stan_function": "gp_transform_irfft",
        "python_function": fft.transform_irfft,
        "arg_types": {"n_": "int", "z": "vector[n_]", "loc": "vector[n_]", "cov": "vector[n_]"},
        "arg_values": {"n_": n, "z": z, "loc": loc, "cov": lincov},
        "result_type": "vector[n_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": y,
    })

    # Evaluate the likelihood.
    configs.append({
        "stan_function": "gp_rfft_lpdf",
        "python_function": fft.evaluate_log_prob_rfft,
        "arg_types": {"n_": "int", "y": "vector[n_]", "loc": "vector[n_]", "cov": "vector[n_]"},
        "arg_values": {"n_": n, "y": y, "loc": loc, "cov": lincov},
        "result_type": "real",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": stats.multivariate_normal(loc, cov).logpdf(y),
    })

for n, m in [(5, 7), (5, 8), (6, 7), (6, 8)]:
    # Two-dimensional real Fourier transform ...
    y = np.random.normal(0, 1, (n, m))
    configs.append({
        "stan_function": "rfft2",
        "python_function": np.fft.rfft2,
        "arg_types": {"n_": "int", "m_": "int", "a": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "a": y},
        "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
        "includes": ["gptools_util.stan"],
    })

    # ... and its inverse.
    z = np.fft.rfft2(y)
    configs.append({
        "stan_function": "inv_rfft2",
        "python_function": lambda a, m: np.fft.irfft2(a, (a.shape[0], m)),
        "arg_types": {"n_": "int", "m": "int", "a": "complex_matrix[n_, m %/% 2 + 1]"},
        "arg_values": {"a": z, "n_": n, "m": m},
        "result_type": "matrix[n_, m]",
        "includes": ["gptools_util.stan"],
        "desired": y,
    })

    # Unpack truncated Fourier coefficients to a real vector ...
    configs.append({
        "stan_function": "gp_unpack_rfft2",
        "python_function": lambda z, m: fft.unpack_rfft2(z, (z.shape[0], m)),
        "arg_types": {"n_": "int", "m": "int", "z": "complex_matrix[n_, m %/% 2 + 1]"},
        "arg_values": {"z": z, "n_": n, "m": m},
        "result_type": "matrix[n_, m]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
    })

    # ... and pack them up again.
    unpacked_z = fft.unpack_rfft2(z, (n, m))
    configs.append({
        "stan_function": "gp_pack_rfft2",
        "python_function": fft.pack_rfft2,
        "arg_types": {"n_": "int", "m_": "int", "z": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "z": unpacked_z},
        "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": z,
    })

    # Transforming to whitened Fourier coefficients ...
    loc = np.random.normal(0, 1, (n, m))
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    xs = coordgrid(np.arange(n), np.arange(m))
    cov = kernel(xs)
    lincov = cov[0].reshape((n, m))
    configs.append({
        "stan_function": "gp_transform_rfft2",
        "python_function": fft.transform_rfft2,
        "arg_types": {"n_": "int", "m_": "int", "y": "matrix[n_, m_]", "loc": "matrix[n_, m_]",
                      "cov": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "y": y, "loc": loc, "cov": lincov},
        "result_type": "matrix[n_, m_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
    })

    # Evaluate the likelihood.
    configs.append({
        "stan_function": "gp_rfft2_lpdf",
        "python_function": fft.evaluate_log_prob_rfft2,
        "arg_types": {"n_": "int", "m_": "int", "y": "matrix[n_, m_]", "loc": "matrix[n_, m_]",
                      "cov": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "y": y, "loc": loc, "cov": lincov},
        "result_type": "real",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": stats.multivariate_normal(loc.ravel(), cov).logpdf(y.ravel()),
    })

for ndim in [1, 2, 3]:
    n = 1 + np.random.poisson(50)
    m = 1 + np.random.poisson(50)
    sigma = np.random.gamma(10, 0.1)
    length_scale = np.random.gamma(10, 0.1, ndim)
    period = np.random.gamma(100, 0.1, ndim)
    x = np.random.uniform(0, period, (n, ndim))
    y = np.random.uniform(0, period, (m, ndim))
    configs.append({
        "stan_function": "gp_periodic_exp_quad_cov",
        "python_function": lambda x, y, sigma, length_scale, period: kernels.ExpQuadKernel(
            sigma, length_scale, period=period)(x[:, None], y[None]),
        "arg_types": {"n_": "int", "m_": "int", "p_": "int", "x": "array [n_] vector[p_]",
                      "y": "array [m_] vector[p_]", "sigma": "real", "length_scale": "vector[p_]",
                      "period": "vector[p_]"},
        "arg_values": {"n_": n, "m_": m, "p_": ndim, "x": x, "y": y, "sigma": sigma,
                       "length_scale": length_scale, "period": period},
        "result_type": "matrix[n_, m_]",
        "includes": ["gptools_kernels.stan"],
    })


@pytest.mark.parametrize("config", configs)
def test_stan_python_equal(config: dict) -> None:
    assert_stan_python_allclose(**config)
