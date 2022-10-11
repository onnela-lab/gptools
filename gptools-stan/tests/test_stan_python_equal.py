from gptools.stan import compile_model
import hashlib
import numpy as np
import pathlib
import pytest
import typing


def assert_stan_python_allclose(
        stan_function: str, python_function: typing.Callable, arg_types: dict[str, str],
        arg_values: dict[str, np.ndarray], result_type: str, includes: typing.Iterable[str]) \
            -> None:
    """
    Assert that a Stan and Python function return the same result up to numerical inaccuracies.
    """
    # Assemble the stan code we seek to build.
    functions = "\n".join(f"#include {include}" for include in includes)
    data = "\n".join(f"{type} {name};" for name, type in arg_types.items())
    generated_quantities = f"{result_type} result = {stan_function}(" \
        + ", ".join(arg for arg in arg_values if not arg.endswith("_")) + ");"
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
    model = compile_model(stan_file=path)
    fit = model.sample(arg_values, fixed_param=True, iter_sampling=1, iter_warmup=1, sig_figs=9)
    result_stan, = fit.stan_variable("result")

    # Get the python result and compare.
    result_python = python_function(**{key: value for key, value in arg_values.items() if not
                                       key.endswith("_")})
    np.testing.assert_allclose(result_stan, result_python)


configs = []

for n in [5, 6]:
    x = np.random.normal(0, 1, n)
    configs.append({
        "stan_function": "rfft",
        "python_function": np.fft.rfft,
        "arg_types": {"n_": "int", "a": "vector[n_]"},
        "arg_values": {"n_": n, "a": x},
        "result_type": "complex_vector[n_ %/% 2 + 1]",
        "includes": ["gptools_util.stan"],
    })

    y = np.fft.rfft(x)
    configs.append({
        "stan_function": "inv_rfft",
        "python_function": np.fft.irfft,
        "arg_types": {"n": "int", "a": "complex_vector[n %/% 2 + 1]"},
        "arg_values": {"a": y, "n": n},
        "result_type": "vector[n]",
        "includes": ["gptools_util.stan"],
    })

for n, m in [(5, 7), (5, 8), (6, 7), (6, 8)]:
    x = np.random.normal(0, 1, (n, m))
    configs.append({
        "stan_function": "rfft2",
        "python_function": np.fft.rfft2,
        "arg_types": {"n_": "int", "m_": "int", "a": "matrix[n_, m_]"},
        "arg_values": {"n_": n, "m_": m, "a": x},
        "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
        "includes": ["gptools_util.stan"],
    })

    y = np.fft.rfft2(x)
    configs.append({
        "stan_function": "inv_rfft2",
        "python_function": lambda a, m: np.fft.irfft2(a, (a.shape[0], m)),
        "arg_types": {"n_": "int", "m": "int", "a": "complex_matrix[n_, m %/% 2 + 1]"},
        "arg_values": {"a": y, "n_": n, "m": m},
        "result_type": "matrix[n_, m]",
        "includes": ["gptools_util.stan"],
    })


@pytest.mark.parametrize("config", configs)
def test_stan_python_equal(config: dict) -> None:
    assert_stan_python_allclose(**config)
