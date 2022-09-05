from gptools.util.kernels import ExpQuadKernel
from gptools.torch import GraphGaussianProcess, ParametrizedDistribution, \
    TerminateOnPlateau, VariationalModel
from gptools.util import lattice_predecessors
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
    predecessors = th.as_tensor(lattice_predecessors((n,), k))

    return {
        "n": n,
        "x": x,
        "loc": loc,
        "kernel": kernel,
        "dist": dist,
        "ys": ys,
        "predecessors": predecessors,
    }


# We omit "gels" because masked covariances do not have full rank.
@pytest.mark.parametrize("lstsq_driver", ["gelsy", "gelsd", "gelss"])
def test_torch_evaluate_log_prob(data: dict, lstsq_driver: str) -> None:
    """
    Generate a few datasets and ensure that the log prob using the full covariance is correlated
    with the log prob of the nearest-neighbor based covariance.
    """
    from scipy import stats

    gdist = GraphGaussianProcess(data["loc"], data["x"], data["predecessors"], data["kernel"],
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
        GraphGaussianProcess(data["loc"], data["x"], data["predecessors"], data["kernel"],
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
    predecessors = lattice_predecessors(x.shape, 7)
    dist = GraphGaussianProcess(th.zeros_like(x), X, predecessors, kernel)
    y = dist.sample(size)
    assert y.shape == th.Size(size or ()) + (num_nodes,)


def test_parametrized_distribution() -> None:
    loc = - th.ones(3)
    scale = 1.0 + th.arange(3)
    pdist = ParametrizedDistribution(th.distributions.Normal, loc=loc, scale=scale, const={"loc"})
    dist = pdist()
    assert loc.allclose(dist.loc)
    assert scale.allclose(dist.scale)
    assert not dist.loc.grad_fn
    assert dist.scale.grad_fn


def test_check_log_prob_shape() -> None:
    class Model(VariationalModel):
        def log_prob(self, parameters, reduce):
            log_prob = parameters["x"].sum(axis=-1)
            return log_prob.sum() if reduce else log_prob

    model = Model({
        "x": ParametrizedDistribution(th.distributions.Normal, loc=th.zeros(3), scale=th.ones(3)),
    })
    model.check_log_prob_shape(reduce=False)
    with pytest.raises(RuntimeError):
        model.check_log_prob_shape(reduce=True)


@pytest.mark.parametrize("init, sequence, cont", [
    ({"patience": 3}, [3, 3, 2], True),
    ({"patience": 3}, [3, 3, 3], True),
    ({"patience": 3}, [3, 3, 3, 3], False),
    ({"patience": 10, "max_num_steps": 3}, [1, 2], True),
    ({"patience": 10, "max_num_steps": 3}, [1, 2, 3], False),
])
def test_terminate_on_plateau(init: dict, sequence: list, cont: bool) -> None:
    terminator = TerminateOnPlateau(**init)
    for x in sequence:
        if not terminator.step(x):
            break
    assert bool(terminator) == cont
