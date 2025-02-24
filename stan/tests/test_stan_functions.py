from gptools.stan import compile_model
from gptools.util import coordgrid, fft, kernels
import hashlib
import inspect
import numpy as np
import pathlib
import pytest
from scipy import stats
from typing import Dict, Iterable, List, Optional, Union


CONFIGURATIONS: List[dict] = []


def add_configuration(configuration: dict) -> dict:
    frame = inspect.currentframe().f_back
    configuration.setdefault(
        "line_info", f"{frame.f_code.co_filename}:{frame.f_lineno}"
    )
    CONFIGURATIONS.append(configuration)
    return configuration


def get_configuration_ids() -> Iterable[str]:
    configuration_ids = []
    for configuration in CONFIGURATIONS:
        parts = [configuration["stan_function"]]
        parts.extend(
            str(arg)
            for arg in configuration["arg_values"].values()
            if not isinstance(arg, np.ndarray)
        )
        if suffix := configuration.get("suffix"):
            parts.append(suffix)
        configuration_id = "-".join(parts)
        if configuration_id in configuration_ids:
            raise ValueError(f"configuration id {configuration_id} already exists")
        configuration_ids.append(configuration_id)
    return configuration_ids


def assert_stan_function_allclose(
    stan_function: str,
    arg_types: Dict[str, str],
    arg_values: Dict[str, np.ndarray],
    result_type: str,
    desired: Union[np.ndarray, List[np.ndarray]],
    atol: float = 1e-6,
    includes: Optional[Iterable[str]] = None,
    line_info: Optional[str] = "???",
    suffix: Optional[str] = None,
    raises: bool = False,
) -> None:
    """
    Assert that a Stan and Python function return the same result up to numerical
    inaccuracies.
    """
    # Assemble the stan code we seek to build.
    functions = "\n".join(f"#include {include}" for include in includes or [])
    data = "\n".join(f"{type} {name};" for name, type in arg_types.items())
    args = [arg for arg in arg_values if not arg.endswith("_")]
    if stan_function.endswith("_lpdf"):
        x, *args = args
        generated_quantities = f"{stan_function}({x} | {', '.join(args)});"
    else:
        generated_quantities = f"{stan_function}({', '.join(args)});"
    if result_type:
        generated_quantities = f"{result_type} result = {generated_quantities}"
    code = "\n".join(
        [
            "functions {",
            functions,
            "}",
            "data {",
            data,
            "}",
            "generated quantities {",
            generated_quantities,
            "real success = 1;",
            "}",
        ]
    )

    # Write to file if it does not already exist.
    digest = hashlib.sha256(code.encode()).hexdigest()
    path = pathlib.Path(".cmdstanpy_cache", digest).with_suffix(".stan")
    if not path.is_file():
        path.parent.mkdir(exist_ok=True)
        path.write_text(code)

    # Compile the model and obtain the result.
    try:
        model = compile_model(stan_file=path)
    except Exception as ex:
        raise RuntimeError(
            f"failed to compile model for {stan_function} at {line_info}"
        ) from ex

    try:
        fit = model.sample(
            arg_values,
            fixed_param=True,
            iter_sampling=1,
            iter_warmup=1,
            sig_figs=9,
            chains=1,
        )
        (success,) = fit.stan_variable("success")
        if not success or np.isnan(success):
            console = pathlib.Path(fit.runset.stdout_files[0]).read_text()
            raise RuntimeError(f"failed to sample from model: \n{console}")
    except Exception as ex:
        if raises:
            return
        raise RuntimeError(
            f"failed to get Stan result for {stan_function} at {line_info}"
        ) from ex

    if raises:
        raise RuntimeError(
            f"sampling did not raise an error for {stan_function} at {line_info}"
        )

    # Skip validation if we don't have a result type.
    if not result_type or desired is None:
        return
    # Verify against expected value. We only check one because we have already verified
    # that they are the same.
    result = fit.stan_variable("result")[0]
    if not isinstance(desired, list):
        desired = [desired]
    try:
        for value in desired:
            np.testing.assert_allclose(result, value, atol=atol)
    except Exception as ex:
        raise RuntimeError(
            f"unexpected result for {stan_function} at {line_info}: \n{ex}"
        ) from ex


add_configuration(
    {
        "stan_function": "linspaced_vector",
        "arg_types": {"n": "int", "a": "real", "b": "real"},
        "arg_values": {"n": 10, "a": 0, "b": 9},
        "result_type": "vector[n]",
        "desired": np.arange(10),
    }
)


for n in [7, 8]:
    # One-dimensional real Fourier transform ...
    y = np.random.normal(0, 1, n)
    z = np.fft.rfft(y)
    add_configuration(
        {
            "stan_function": "rfft",
            "arg_types": {"n_": "int", "a": "vector[n_]"},
            "arg_values": {"n_": n, "a": y},
            "result_type": "complex_vector[n_ %/% 2 + 1]",
            "includes": ["gptools/util.stan"],
            "desired": z,
        }
    )

    # ... and its inverse.
    add_configuration(
        {
            "stan_function": "inv_rfft",
            "arg_types": {"n": "int", "a": "complex_vector[n %/% 2 + 1]"},
            "arg_values": {"a": z, "n": n},
            "result_type": "vector[n]",
            "includes": ["gptools/util.stan"],
            "desired": [np.fft.irfft(z, n), y],
        }
    )

    # Unpack truncated Fourier coefficients to a real vector ...
    unpacked_z = fft.fft1.unpack_rfft(z, n)
    add_configuration(
        {
            "stan_function": "gp_unpack_rfft",
            "arg_types": {"size": "int", "z": "complex_vector[size %/% 2 + 1]"},
            "arg_values": {"z": z, "size": n},
            "result_type": "vector[size]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan"],
            "desired": unpacked_z,
        }
    )

    # ... and pack them up again.
    add_configuration(
        {
            "stan_function": "gp_pack_rfft",
            "arg_types": {"n_": "int", "z": "vector[n_]"},
            "arg_values": {"n_": n, "z": unpacked_z},
            "result_type": "complex_vector[n_ %/% 2 + 1]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan"],
            "desired": [z, fft.fft1.pack_rfft(unpacked_z)],
        }
    )

    # Transforming to whitened Fourier coefficients ...
    loc = np.random.normal(0, 1, n)
    kernel = kernels.ExpQuadKernel(
        np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1
    ) + kernels.DiagonalKernel(0.1, 1)
    cov = kernel.evaluate(np.arange(n)[:, None])
    lincov = cov[0]
    cov_rfft = np.fft.rfft(lincov).real
    z = fft.transform_rfft(y, loc, cov_rfft=cov_rfft)
    add_configuration(
        {
            "stan_function": "gp_rfft",
            "arg_types": {
                "n_": "int",
                "y": "vector[n_]",
                "loc": "vector[n_]",
                "cov_rfft": "vector[n_ %/% 2 + 1]",
            },
            "arg_values": {"n_": n, "y": y, "loc": loc, "cov_rfft": cov_rfft},
            "result_type": "vector[n_]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan"],
            "desired": z,
        }
    )

    # ... and back again.
    add_configuration(
        {
            "stan_function": "gp_inv_rfft",
            "arg_types": {
                "n_": "int",
                "z": "vector[n_]",
                "loc": "vector[n_]",
                "cov_rfft": "vector[n_ %/% 2 + 1]",
            },
            "arg_values": {"n_": n, "z": z, "loc": loc, "cov_rfft": cov_rfft},
            "result_type": "vector[n_]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan"],
            "desired": [y, fft.transform_irfft(z, loc, cov_rfft=cov_rfft)],
        }
    )

    # Evaluate the likelihood.
    add_configuration(
        {
            "stan_function": "gp_rfft_lpdf",
            "arg_types": {
                "n_": "int",
                "y": "vector[n_]",
                "loc": "vector[n_]",
                "cov_rfft": "vector[n_ %/% 2 + 1]",
            },
            "arg_values": {"n_": n, "y": y, "loc": loc, "cov_rfft": cov_rfft},
            "result_type": "real",
            "includes": ["gptools/util.stan", "gptools/fft1.stan"],
            "desired": [
                fft.evaluate_log_prob_rfft(y, loc, cov_rfft=cov_rfft),
                stats.multivariate_normal(loc, cov).logpdf(y),
            ],
        }
    )

for n, m in [(5, 7), (5, 8), (6, 7), (6, 8)]:
    # Two-dimensional real Fourier transform ...
    y = np.random.normal(0, 1, (n, m))
    z = np.fft.rfft2(y)
    add_configuration(
        {
            "stan_function": "rfft2",
            "arg_types": {"n_": "int", "m_": "int", "a": "matrix[n_, m_]"},
            "arg_values": {"n_": n, "m_": m, "a": y},
            "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
            "includes": ["gptools/util.stan"],
            "desired": z,
        }
    )

    # ... and its inverse.
    add_configuration(
        {
            "stan_function": "inv_rfft2",
            "arg_types": {
                "n_": "int",
                "m": "int",
                "a": "complex_matrix[n_, m %/% 2 + 1]",
            },
            "arg_values": {"a": z, "n_": n, "m": m},
            "result_type": "matrix[n_, m]",
            "includes": ["gptools/util.stan"],
            "desired": [y, np.fft.irfft2(z, (n, m))],
        }
    )

    # Unpack truncated Fourier coefficients to a real vector ...
    unpacked_z = fft.fft2.unpack_rfft2(z, (n, m))
    add_configuration(
        {
            "stan_function": "gp_unpack_rfft2",
            "arg_types": {
                "n_": "int",
                "m": "int",
                "z": "complex_matrix[n_, m %/% 2 + 1]",
            },
            "arg_values": {"z": z, "n_": n, "m": m},
            "result_type": "matrix[n_, m]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan", "gptools/fft2.stan"],
            "desired": unpacked_z,
        }
    )

    # ... and pack them up again.
    add_configuration(
        {
            "stan_function": "gp_pack_rfft2",
            "arg_types": {"n_": "int", "m_": "int", "z": "matrix[n_, m_]"},
            "arg_values": {"n_": n, "m_": m, "z": unpacked_z},
            "result_type": "complex_matrix[n_, m_ %/% 2 + 1]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan", "gptools/fft2.stan"],
            "desired": [z, fft.fft2.pack_rfft2(unpacked_z)],
        }
    )

    # Transforming to whitened Fourier coefficients ...
    loc = np.random.normal(0, 1, (n, m))
    kernel = kernels.ExpQuadKernel(
        np.random.gamma(10, 0.1), np.random.gamma(10, 0.01), 1
    ) + kernels.DiagonalKernel(0.1, 1)
    xs = coordgrid(np.arange(n), np.arange(m))
    cov = kernel.evaluate(xs)
    lincov = cov[0].reshape((n, m))
    cov_rfft2 = np.fft.rfft2(lincov).real
    z = fft.transform_rfft2(y, loc, cov_rfft2=cov_rfft2)
    add_configuration(
        {
            "stan_function": "gp_rfft2",
            "arg_types": {
                "n_": "int",
                "m_": "int",
                "y": "matrix[n_, m_]",
                "loc": "matrix[n_, m_]",
                "cov_rfft2": "matrix[n_, m_ %/% 2 + 1]",
            },
            "arg_values": {
                "n_": n,
                "m_": m,
                "y": y,
                "loc": loc,
                "cov_rfft2": cov_rfft2,
            },
            "result_type": "matrix[n_, m_]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan", "gptools/fft2.stan"],
            "desired": z,
        }
    )

    # ... and back again.
    add_configuration(
        {
            "stan_function": "gp_inv_rfft2",
            "arg_types": {
                "n_": "int",
                "m_": "int",
                "z": "matrix[n_, m_]",
                "loc": "matrix[n_, m_]",
                "cov_rfft2": "matrix[n_, m_ %/% 2 + 1]",
            },
            "arg_values": {
                "n_": n,
                "m_": m,
                "z": z,
                "loc": loc,
                "cov_rfft2": cov_rfft2,
            },
            "result_type": "matrix[n_, m_]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan", "gptools/fft2.stan"],
            "desired": [y, fft.transform_irfft2(z, loc, cov_rfft2=cov_rfft2)],
        }
    )

    # Evaluate the likelihood.
    add_configuration(
        {
            "stan_function": "gp_rfft2_lpdf",
            "arg_types": {
                "n_": "int",
                "m_": "int",
                "y": "matrix[n_, m_]",
                "loc": "matrix[n_, m_]",
                "cov_rfft2": "matrix[n_, m_ %/% 2 + 1]",
            },
            "arg_values": {
                "n_": n,
                "m_": m,
                "y": y,
                "loc": loc,
                "cov_rfft2": cov_rfft2,
            },
            "result_type": "real",
            "includes": ["gptools/util.stan", "gptools/fft1.stan", "gptools/fft2.stan"],
            "desired": [
                stats.multivariate_normal(loc.ravel(), cov).logpdf(y.ravel()),
                fft.evaluate_log_prob_rfft2(y, loc, cov_rfft2=cov_rfft2),
            ],
        }
    )

for m in [7, 8]:
    sigma = np.random.gamma(10, 0.1)
    length_scale = np.random.gamma(10, 0.1)
    period = np.random.gamma(100, 0.1)
    add_configuration(
        {
            "stan_function": "gp_periodic_exp_quad_cov_rfft",
            "arg_types": {
                "m": "int",
                "sigma": "real",
                "length_scale": "real",
                "period": "real",
            },
            "arg_values": {
                "m": n,
                "sigma": sigma,
                "length_scale": length_scale,
                "period": period,
            },
            "result_type": "vector[m %/% 2 + 1]",
            "includes": ["gptools/util.stan", "gptools/fft1.stan"],
            "desired": kernels.ExpQuadKernel(
                sigma, length_scale, period=period
            ).evaluate_rfft([n]),
        }
    )
    for dof in [3 / 2, 5 / 2]:
        add_configuration(
            {
                "stan_function": "gp_periodic_matern_cov_rfft",
                "arg_types": {
                    "dof": "real",
                    "m": "int",
                    "sigma": "real",
                    "length_scale": "real",
                    "period": "real",
                },
                "arg_values": {
                    "dof": dof,
                    "m": n,
                    "sigma": sigma,
                    "length_scale": length_scale,
                    "period": period,
                },
                "result_type": "vector[m %/% 2 + 1]",
                "includes": ["gptools/util.stan", "gptools/fft1.stan"],
                "desired": kernels.MaternKernel(
                    dof, sigma, length_scale, period
                ).evaluate_rfft([n]),
            }
        )
    for n in [9, 10]:
        length_scale = np.random.gamma(10, 0.1, 2)
        period = np.random.gamma(100, 0.1, 2)
        add_configuration(
            {
                "stan_function": "gp_periodic_exp_quad_cov_rfft2",
                "arg_types": {
                    "m": "int",
                    "n": "int",
                    "sigma": "real",
                    "length_scale": "vector[2]",
                    "period": "vector[2]",
                },
                "arg_values": {
                    "m": m,
                    "n": n,
                    "sigma": sigma,
                    "length_scale": length_scale,
                    "period": period,
                },
                "result_type": "matrix[m, n %/% 2 + 1]",
                "includes": ["gptools/util.stan", "gptools/fft.stan"],
                "desired": kernels.ExpQuadKernel(
                    sigma, length_scale, period=period
                ).evaluate_rfft([m, n]),
            }
        )
        for dof in [3 / 2, 5 / 2]:
            kernel = kernels.MaternKernel(dof, sigma, length_scale, period)
            add_configuration(
                {
                    "stan_function": "gp_periodic_matern_cov_rfft2",
                    "arg_types": {
                        "dof": "real",
                        "m": "int",
                        "n": "int",
                        "sigma": "real",
                        "length_scale": "vector[2]",
                        "period": "vector[2]",
                    },
                    "arg_values": {
                        "dof": dof,
                        "m": m,
                        "n": n,
                        "sigma": sigma,
                        "length_scale": length_scale,
                        "period": period,
                    },
                    "result_type": "matrix[m, n %/% 2 + 1]",
                    "includes": ["gptools/util.stan", "gptools/fft.stan"],
                    "desired": kernel.evaluate_rfft([m, n]),
                }
            )

for num_nodes, edges, raises in [
    (2, [np.ones(2), np.zeros(2)], True),  # Successors start at zero.
    (3, [[1, 1], [3, 2]], True),  # Successors are not ordered.
    (2, [[2], [1]], True),  # Predecessors > successors.
    (1, [[1], [1]], True),  # Self-loop.
    (5, [[1, 2], [2, 3]], False),  # Ok.
    (3, [[1, 2, 1], [2, 3, 3]], False),  # Ok.
    (
        7,
        [[1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5], [2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]],
        False,
    ),  # Ok.
]:
    edges = np.asarray(edges)
    add_configuration(
        {
            "stan_function": "out_degrees",
            "arg_types": {
                "num_nodes": "int",
                "num_edges_": "int",
                "edges": "array [2, num_edges_] int",
            },
            "arg_values": {
                "num_nodes": num_nodes,
                "num_edges_": edges.shape[1],
                "edges": edges,
            },
            "result_type": "array [num_nodes] int",
            "includes": ["gptools/util.stan", "gptools/graph.stan"],
            "raises": raises,
            "desired": [],
        }
    )


# Add evaluation of likelihood on a complete graph to check values.
n = 10
for p in [1, 2]:
    for kernel in [
        kernels.ExpQuadKernel(1.3, 0.7 * np.ones(p).squeeze()),
        kernels.MaternKernel(1.5, 1.3, 0.7 * np.ones(p).squeeze()),
        kernels.MaternKernel(2.5, 1.3, 0.7 * np.ones(p).squeeze()),
    ]:
        x = np.random.normal(0, 1, (n, p))
        epsilon = 1e-3
        loc = np.random.normal(0, 1, n)
        cov = kernel.evaluate(x) + epsilon * np.eye(n)
        dist = stats.multivariate_normal(loc, cov)
        y = dist.rvs()
        z = np.random.normal(0, 1, y.shape)
        # Construct a complete graph.
        edges = []
        for i in range(1, n):
            edges.append(np.transpose([np.roll(np.arange(i), 1), np.ones(i) * i]))
        edges = np.concatenate(edges, axis=0).astype(int).T

        length_scale_type = "real" if p == 1 else "array [p_] real"

        # Evaluate the centered parametrization log likelihood.
        if isinstance(kernel, kernels.ExpQuadKernel):
            lpdf_stan_function = "gp_graph_exp_quad_cov_lpdf"
            transform_stan_function = "gp_inv_graph_exp_quad_cov"
        elif isinstance(kernel, kernels.MaternKernel):
            if kernel.dof == 1.5:
                lpdf_stan_function = "gp_graph_matern32_cov_lpdf"
                transform_stan_function = "gp_inv_graph_matern32_cov"
            elif kernel.dof == 2.5:
                lpdf_stan_function = "gp_graph_matern52_cov_lpdf"
                transform_stan_function = "gp_inv_graph_matern52_cov"
            else:
                raise ValueError(kernel.dof)
        else:
            raise TypeError(kernel)

        # Evaluate the log likelihood.
        add_configuration(
            {
                "stan_function": lpdf_stan_function,
                "arg_types": {
                    "p_": "int",
                    "num_nodes_": "int",
                    "num_edges_": "int",
                    "y": "vector[num_nodes_]",
                    "x": "array [num_nodes_] vector[p_]",
                    "sigma": "real",
                    "length_scale": length_scale_type,
                    "edges": "array [2, num_edges_] int",
                    "degrees": "array [num_nodes_] int",
                    "epsilon": "real",
                    "loc": "vector[num_nodes_]",
                },
                "arg_values": {
                    "p_": p,
                    "num_nodes_": n,
                    "num_edges_": edges.shape[1],
                    "y": y,
                    "loc": loc,
                    "x": x,
                    "sigma": kernel.sigma,
                    "length_scale": kernel.length_scale,
                    "edges": edges + 1,
                    "degrees": np.bincount(edges[1], minlength=n),
                    "epsilon": epsilon,
                },
                "result_type": "real",
                "includes": ["gptools/util.stan", "gptools/graph.stan"],
                "desired": dist.logpdf(y),
            }
        )

        # Evaluate the non-centered transformation.
        add_configuration(
            {
                "stan_function": transform_stan_function,
                "arg_types": {
                    "p_": "int",
                    "num_nodes_": "int",
                    "num_edges_": "int",
                    "z": "vector[num_nodes_]",
                    "x": "array [num_nodes_] vector[p_]",
                    "sigma": "real",
                    "length_scale": length_scale_type,
                    "edges": "array [2, num_edges_] int",
                    "loc": "vector[num_nodes_]",
                },
                "arg_values": {
                    "p_": p,
                    "num_nodes_": n,
                    "num_edges_": edges.shape[1],
                    "z": z,
                    "loc": loc,
                    "x": x,
                    "sigma": kernel.sigma,
                    "length_scale": kernel.length_scale,
                    "edges": edges + 1,
                },
                "result_type": "vector[num_nodes_]",
                "includes": ["gptools/util.stan", "gptools/graph.stan"],
                "desired": None,  # Only check we can execute, not the result.
            }
        )


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=get_configuration_ids())
def test_stan_function(config: dict) -> None:
    assert_stan_function_allclose(**config)
