from graph_gaussian_process.kernels import ExpQuadKernel
from graph_gaussian_process.torch import GraphGaussianProcess
from graph_gaussian_process.util import lattice_neighborhoods
import itertools as it
import pytest
import torch as th


@pytest.fixture
def data() -> dict:
    n = 50
    x = th.linspace(0, 1, n)[:, None]
    loc = 2 - th.linspace(0, 1, n) ** 2
    kernel = ExpQuadKernel(1, 0.1, 1e-3)
    cov = kernel(x)

    # Obtain the full distribution and graph-based distribution.
    dist = th.distributions.MultivariateNormal(loc, cov)
    ys = dist.sample([100])

    k = 10
    neighborhoods = th.as_tensor(lattice_neighborhoods((n,), k))

    return {
        "n": n,
        "x": x,
        "loc": loc,
        "kernel": kernel,
        "dist": dist,
        "ys": ys,
        "neighborhoods": neighborhoods,
    }


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
# We omit "gels" because masked covariances do not have full rank.
@pytest.mark.parametrize("lstsq_driver", ["gelsy", "gelsd", "gelss"])
def test_torch_evaluate_log_prob(data: dict, lstsq_driver: str) -> None:
    """
    Generate a few datasets and ensure that the log prob using the full covariance is correlated
    with the log prob of the nearest-neighbor based covariance.
    """
    from scipy import stats

    gdist = GraphGaussianProcess(data["loc"], data["x"], data["neighborhoods"], data["kernel"],
                                 lstsq_driver=lstsq_driver)
    # Evaluate log probabilities and ensure they are correlated with the full log probabilities.
    corr, pval = stats.pearsonr(data["dist"].log_prob(data["ys"]), gdist.log_prob(data["ys"]))
    assert corr > 0.9 and pval < 1e-3

    # Test for determinism by evaluating the log probability many times and comparing.
    log_probs = [gdist.log_prob(data["ys"]) for _ in range(100)]
    for a, b in it.pairwise(log_probs):
        assert a.allclose(b)


@pytest.mark.parametrize("lstsq_drivers", it.product(*[["gelsy", "gelsd", "gelss"]] * 2))
def test_torch_evaluate_log_prob_drivers(data: dict, lstsq_drivers: str) -> None:
    a, b = [
        GraphGaussianProcess(data["loc"], data["x"], data["neighborhoods"], data["kernel"],
                             lstsq_driver=driver).log_prob(data["ys"]) for driver in lstsq_drivers
    ]
    if "gelsy" in lstsq_drivers:
        pytest.xfail("https://github.com/pytorch/pytorch/issues/71222")
    mrd = (2 * (a - b) / (a + b)).abs().max()
    assert th.allclose(a, b, atol=1e-4, rtol=1e-4), \
        f"drivers {lstsq_drivers} do not yield same log probs; max relative difference: {mrd}"


@pytest.mark.parametrize("size", [None, [1], [3, 4]])
def test_torch_sample(size: th.Size) -> None:
    num_nodes = 100
    x = th.linspace(0, 1, num_nodes)
    X = x[:, None]
    kernel = ExpQuadKernel(1.4, 0.1, 1e-3)
    neighborhoods = lattice_neighborhoods(x.shape, 7)
    dist = GraphGaussianProcess(th.zeros_like(x), X, neighborhoods, kernel)
    y = dist.sample(size)
    assert y.shape == th.Size(size or ()) + (num_nodes,)
