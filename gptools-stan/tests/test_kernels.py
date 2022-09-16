from gptools.stan import compile_model
from gptools.util import kernels
from gptools.util.testing import KernelConfiguration
import numpy as np
import pathlib
import pytest


def test_kernel_equivalence(kernel_configuration: KernelConfiguration) -> None:
    kernel: kernels.ExpQuadKernel = kernel_configuration()
    if not kernel.is_periodic:
        pytest.skip("kernel is not periodic")

    stan_file = pathlib.Path(__file__).parent / "test_gp_period_exp_quad_cov.stan"
    gp_periodic_exp_quad_cov_model = compile_model(stan_file=stan_file)

    # TODO: use different number of samples once the issue with nugget variance is fixed.
    n = m = 17
    x1 = kernel_configuration.sample_locations(n)
    x2 = kernel_configuration.sample_locations(m)
    _, p = x1.shape
    data = {
        "n": n, "m": m, "p": p, "x1": x1, "x2": x2, "sigma": kernel.alpha,
        "length_scale": kernel.rho * np.ones(p), "period": kernel.period, "epsilon": kernel.epsilon,
    }
    fit = gp_periodic_exp_quad_cov_model.sample(data, iter_sampling=1, iter_warmup=1,
                                                fixed_param=True, sig_figs=9)
    np.testing.assert_allclose(fit.stan_variable("cov")[0], kernel(x1[:, None], x2[None]))
