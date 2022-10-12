from gptools.stan import compile_model
from gptools.util import coordgrid, fft, kernels
import hashlib
import inspect
import numpy as np
import pathlib
import pytest
from scipy import stats
import typing


CONFIGURATIONS = []


def add_configuration(configuration: dict) -> dict:
    frame = inspect.currentframe().f_back
    configuration.setdefault("line_info", f"{frame.f_code.co_filename}:{frame.f_lineno}")
    CONFIGURATIONS.append(configuration)
    return configuration


def assert_stan_python_allclose(
        stan_function: str, arg_types: dict[str, str], arg_values: dict[str, np.ndarray],
        result_type: str, desired: typing.Union[np.ndarray, list[np.ndarray]], atol: float = 1e-8,
        includes: typing.Optional[typing.Iterable[str]] = None,
        line_info: typing.Optional[str] = "???") -> None:
    """
    Assert that a Stan and Python function return the same result up to numerical inaccuracies.
    """
    # Assemble the stan code we seek to build.
    functions = "\n".join(f"#include {include}" for include in includes or [])
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
        result, = fit.stan_variable("result")
    except Exception as ex:
        raise RuntimeError(f"failed to get Stan result for {stan_function} at {line_info}") from ex

    # Verify against expected value. We only check one because we have already verified that they
    # are the same.
    if not isinstance(desired, list):
        desired = [desired]
    try:
        for value in desired:
            np.testing.assert_allclose(result, value, atol=atol)
    except Exception as ex:
        raise RuntimeError(f"unexpected result for {stan_function} at {line_info}") from ex


for n in [7, 8]:
    # One-dimensional real Fourier transform ...
    y = np.random.normal(0, 1, n)
    z = np.fft.rfft(y)
    add_configuration({
        "stan_function": "rfft",
        "arg_types": {"n_": "int", "a": "vector[n_]"},
        "arg_values": {"n_": n, "a": y},
        "result_type": "complex_vector[n_ %/% 2 + 1]",
        "includes": ["gptools_util.stan"],
        "desired": z,
    })

    # ... and its inverse.
    add_configuration({
        "stan_function": "inv_rfft",
        "arg_types": {"n": "int", "a": "complex_vector[n %/% 2 + 1]"},
        "arg_values": {"a": z, "n": n},
        "result_type": "vector[n]",
        "includes": ["gptools_util.stan"],
        "desired": [np.fft.irfft(z, n), y],
    })

    # Unpack truncated Fourier coefficients to a real vector ...
    unpacked_z = fft.unpack_rfft(z, n)
    add_configuration({
        "stan_function": "gp_unpack_rfft",
        "arg_types": {"size": "int", "z": "complex_vector[size %/% 2 + 1]"},
        "arg_values": {"z": z, "size": n},
        "result_type": "vector[size]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": unpacked_z,
    })

    # ... and pack them up again.
    add_configuration({
        "stan_function": "gp_pack_rfft",
        "arg_types": {"n_": "int", "z": "vector[n_]"},
        "arg_values": {"n_": n, "z": unpacked_z},
        "result_type": "complex_vector[n_ %/% 2 + 1]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": [z, fft.pack_rfft(unpacked_z)],
    })

    # Transforming to whitened Fourier coefficients ...
    loc = np.random.normal(0, 1, n)
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    cov = kernel(np.arange(n)[:, None])
    lincov = cov[0]
    z = fft.transform_rfft(y, loc, lincov)
    add_configuration({
        "stan_function": "gp_transform_rfft",
        "arg_types": {"n_": "int", "y": "vector[n_]", "loc": "vector[n_]", "cov": "vector[n_]"},
        "arg_values": {"n_": n, "y": y, "loc": loc, "cov": lincov},
        "result_type": "vector[n_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": z,
    })

    # ... and back again.
    add_configuration({
        "stan_function": "gp_transform_irfft",
        "arg_types": {"n_": "int", "z": "vector[n_]", "loc": "vector[n_]", "cov": "vector[n_]"},
        "arg_values": {"n_": n, "z": z, "loc": loc, "cov": lincov},
        "result_type": "vector[n_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": [y, fft.transform_irfft(z, loc, lincov)],
    })

    # Evaluate the likelihood.
    add_configuration({
        "stan_function": "gp_rfft_lpdf",
        "arg_types": {"n_": "int", "y": "vector[n_]", "loc": "vector[n_]", "cov": "vector[n_]"},
        "arg_values": {"n_": n, "y": y, "loc": loc, "cov": lincov},
        "result_type": "real",
        "includes": ["gptools_util.stan", "gptools_fft1.stan"],
        "desired": [fft.evaluate_log_prob_rfft(y, loc, lincov),
                    stats.multivariate_normal(loc, cov).logpdf(y)],
    })

    # Containers.
    for x in [np.zeros, np.ones]:
        add_configuration({
            "stan_function": x.__name__,
            "arg_types": {"n": "int"},
            "arg_values": {"n": n},
            "result_type": "vector[n]",
            "includes": ["gptools_util.stan"],
            "desired": x(n),
        })

for n, m in [(5, 7), (5, 8), (6, 7), (6, 8)]:
    # Two-dimensional real Fourier transform ...
    y = np.random.normal(0, 1, (n, m))
    z = np.fft.rfft2(y)
    add_configuration({
        "stan_function": "rfft2",
        "arg_types": {"n_": "int", "m_": "int", "a": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "a": y},
        "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
        "includes": ["gptools_util.stan"],
        "desired": z,
    })

    # ... and its inverse.
    add_configuration({
        "stan_function": "inv_rfft2",
        "arg_types": {"n_": "int", "m": "int", "a": "complex_matrix[n_, m %/% 2 + 1]"},
        "arg_values": {"a": z, "n_": n, "m": m},
        "result_type": "matrix[n_, m]",
        "includes": ["gptools_util.stan"],
        "desired": [y, np.fft.irfft2(z, (n, m))],
    })

    # Unpack truncated Fourier coefficients to a real vector ...
    unpacked_z = fft.unpack_rfft2(z, (n, m))
    add_configuration({
        "stan_function": "gp_unpack_rfft2",
        "arg_types": {"n_": "int", "m": "int", "z": "complex_matrix[n_, m %/% 2 + 1]"},
        "arg_values": {"z": z, "n_": n, "m": m},
        "result_type": "matrix[n_, m]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": unpacked_z,
    })

    # ... and pack them up again.
    add_configuration({
        "stan_function": "gp_pack_rfft2",
        "arg_types": {"n_": "int", "m_": "int", "z": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "z": unpacked_z},
        "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": [z, fft.pack_rfft2(unpacked_z)],
    })

    # Transforming to whitened Fourier coefficients ...
    loc = np.random.normal(0, 1, (n, m))
    kernel = kernels.ExpQuadKernel(np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 0.1, 1)
    xs = coordgrid(np.arange(n), np.arange(m))
    cov = kernel(xs)
    lincov = cov[0].reshape((n, m))
    z = fft.transform_rfft2(y, loc, lincov)
    add_configuration({
        "stan_function": "gp_transform_rfft2",
        "arg_types": {"n_": "int", "m_": "int", "y": "matrix[n_, m_]", "loc": "matrix[n_, m_]",
                      "cov": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "y": y, "loc": loc, "cov": lincov},
        "result_type": "matrix[n_, m_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": z,
    })

    # ... and back again.
    add_configuration({
        "stan_function": "gp_transform_irfft2",
        "arg_types": {"n_": "int", "m_": "int", "z": "matrix[n_, m_]", "loc": "matrix[n_, m_]",
                      "cov": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "z": z, "loc": loc, "cov": lincov},
        "result_type": "matrix[n_, m_]",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": [y, fft.transform_irfft2(z, loc, lincov)],
    })

    # Evaluate the likelihood.
    add_configuration({
        "stan_function": "gp_rfft2_lpdf",
        "arg_types": {"n_": "int", "m_": "int", "y": "matrix[n_, m_]", "loc": "matrix[n_, m_]",
                      "cov": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "y": y, "loc": loc, "cov": lincov},
        "result_type": "real",
        "includes": ["gptools_util.stan", "gptools_fft1.stan", "gptools_fft2.stan"],
        "desired": [stats.multivariate_normal(loc.ravel(), cov).logpdf(y.ravel()),
                    fft.evaluate_log_prob_rfft2(y, loc, lincov)],
    })

    # Containers.
    for x in [np.zeros, np.ones]:
        add_configuration({
            "stan_function": x.__name__,
            "arg_types": {"n": "int", "m": "int"},
            "arg_values": {"n": n, "m": m},
            "result_type": "matrix[n, m]",
            "includes": ["gptools_util.stan"],
            "desired": x((n, m)),
        })

    # Using `to_vector` is different from numpy's `ravel` in terms of ordering ...
    add_configuration({
        "stan_function": "to_vector",
        "arg_types": {"n_": "int", "m_": "int", "y": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "y": y},
        "result_type": "vector[n_ * m_]",
        "includes": ["gptools_util.stan"],
        "desired": y.T.ravel(),
    })

    # ... but `to_array_1d` has the same ordering as numpy's `ravel`.
    add_configuration({
        "stan_function": "to_array_1d",
        "arg_types": {"n_": "int", "m_": "int", "y": "array [n_, m_] real"},
        "arg_values": {"n_": n, "m_": m, "y": y},
        "result_type": "array [n_ * m_] real",
        "includes": ["gptools_util.stan"],
        "desired": y.ravel(),
    })

    # Linearising matrices and arrays using a common `ravel` function.
    add_configuration({
        "stan_function": "ravel",
        "arg_types": {"n_": "int", "m_": "int", "y": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "y": y},
        "result_type": "vector[n_ * m_]",
        "includes": ["gptools_util.stan"],
        "desired": y.ravel(),
    })
    add_configuration({
        "stan_function": "ravel",
        "arg_types": {"n_": "int", "m_": "int", "y": "array [n_, m_] real"},
        "arg_values": {"n_": n, "m_": m, "y": y},
        "result_type": "array [n_ * m_] real",
        "includes": ["gptools_util.stan"],
        "desired": y.ravel(),
    })

for ndim in [1, 2, 3]:
    n = 1 + np.random.poisson(50)
    m = 1 + np.random.poisson(50)
    sigma = np.random.gamma(10, 0.1)
    length_scale = np.random.gamma(10, 0.1, ndim)
    period = np.random.gamma(100, 0.1, ndim)
    x = np.random.uniform(0, period, (n, ndim))
    y = np.random.uniform(0, period, (m, ndim))
    add_configuration({
        "stan_function": "gp_periodic_exp_quad_cov",
        "arg_types": {"n_": "int", "m_": "int", "p_": "int", "x": "array [n_] vector[p_]",
                      "y": "array [m_] vector[p_]", "sigma": "real", "length_scale": "vector[p_]",
                      "period": "vector[p_]"},
        "arg_values": {"n_": n, "m_": m, "p_": ndim, "x": x, "y": y, "sigma": sigma,
                       "length_scale": length_scale, "period": period},
        "result_type": "matrix[n_, m_]",
        "includes": ["gptools_kernels.stan"],
        "desired": kernels.ExpQuadKernel(sigma, length_scale, period=period)(x[:, None], y[None]),
    })


@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_stan_python_equal(config: dict) -> None:
    assert_stan_python_allclose(**config)
