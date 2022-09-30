from gptools.torch.util import ParameterizedDistribution, TerminateOnPlateau, VariationalModel
import pytest
import torch as th


def test_parameterized_distribution() -> None:
    loc = - th.ones(3)
    scale = 1.0 + th.arange(3)
    pdist = ParameterizedDistribution(th.distributions.Normal, loc=loc, scale=scale, const={"loc"})
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
        "x": ParameterizedDistribution(th.distributions.Normal, loc=th.zeros(3), scale=th.ones(3)),
    })
    model.check_log_prob_shape(reduce=False)
    with pytest.raises(RuntimeError):
        model.check_log_prob_shape(reduce=True)


def test_batch_elbo_estimate() -> None:
    class Model(VariationalModel):
        def log_prob(self, parameters):
            return parameters["x"].sum(axis=-1)

    model = Model({
        "x": ParameterizedDistribution(th.distributions.Normal, loc=th.zeros(3), scale=th.ones(3)),
    })
    assert model.batch_elbo_estimate((2, 3)).shape == (2, 3)


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
