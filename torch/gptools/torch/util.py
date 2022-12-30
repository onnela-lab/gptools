from __future__ import annotations
import torch as th
from torch.distributions import constraints
from typing import Optional


DistributionDict = dict[str, th.distributions.Distribution]
TensorDict = dict[str, th.Tensor]


class ParameterizedDistribution(th.nn.Module):
    """
    Parameterized distribution with initial conditions. Parameters are transformed to an
    unconstrained space for optimization, and the distribution is constructed by transforming back
    to the constrained space in the forward pass.

    Args:
        cls: Distribution to create.
        const: Sequence of parameter names that should be treated as constants, i.e., not require
            gradients.
        **kwargs: Initial conditions of the distribution.
    """
    def __init__(self, cls: type[th.distributions.Distribution], *,
                 const: Optional[set[str]] = None, **kwargs: TensorDict):
        super().__init__()
        const = const or set()
        self.cls = cls
        unconstrained = {}
        self.transforms = {}
        self.const = {}
        for key, value in kwargs.items():
            constraint: constraints.Constraint = cls.arg_constraints.get(key)
            if constraint is None or key in const:
                self.const[key] = value
            else:
                transform: th.distributions.Transform = th.distributions.transform_to(constraint)
                unconstrained[key] = th.nn.Parameter(transform.inv(th.as_tensor(value)))
                self.transforms[key] = transform
        self.unconstrained = th.nn.ParameterDict(unconstrained)

    def forward(self) -> th.distributions.Distribution:
        kwargs = {key: self.transforms[key](value) for key, value in self.unconstrained.items()}
        kwargs.update(self.const)
        return self.cls(**kwargs)


class VariationalModel(th.nn.Module):
    """
    Base class for variational posterior approximations.

    Args:
        approximations: Variational factors approximating the posterior.
    """
    def __init__(self, approximations: dict[str, ParameterizedDistribution]) -> None:
        super().__init__()
        self.approximations = th.nn.ModuleDict(approximations)

    def log_prob(self, parameters: TensorDict, *args, **kwargs) -> th.Tensor:
        """
        Evaluate the log joint probability of the model.

        Args:
            parameters: Parameters of the model.

        Returns:
            log_prob: Log joint probability.
        """
        raise NotImplementedError

    def distributions(self) -> DistributionDict:
        """
        Get distribution instances of the variational approximations.
        """
        return {key: value() for key, value in self.approximations.items()}

    def rsample(self, size: Optional[th.Size] = None,
                distributions: Optional[DistributionDict] = None) -> TensorDict:
        """
        Draw reparameterized samples from the variational approximations.
        """
        distributions = distributions or self.distributions()
        size = () if size is None else size
        return {key: distribution.rsample(size) for key, distribution in distributions.items()}

    def entropies(self, distributions: Optional[DistributionDict] = None) -> TensorDict:
        """
        Get entropies of each variational approximation.

        Args:
            distributions: Variational factors whose entropy to evaluate; defaults to
                :meth:`distributions`.

        Returns:
            entropy: Entropies of variational factors keyed by parameter name.
        """
        distributions = distributions or self.distributions()
        return {key: distribution.entropy() for key, distribution in distributions.items()}

    def entropy(self, distributions: Optional[DistributionDict] = None) -> th.Tensor:
        """
        Get entropy of the joint variational approximation.

        Args:
            distributions: Variational factors whose entropy to evaluate; defaults to
                :meth:`distributions`.

        Returns:
            entropy: Entropy of the joint variational approximation as a scalar.
        """
        return sum(entropy.sum() for entropy in self.entropies(distributions).values())

    def elbo_estimate(self, parameters: TensorDict, *args,
                      distributions: Optional[DistributionDict] = None, **kwargs) -> th.Tensor:
        """
        Evaluate an estimate of the evidence lower bound given parameter values.

        Args:
            parameters: Sample of parameters drawn from the variational approximation.
            distributions: Variational factors whose entropy to evaluate; defaults to
                :meth:`distributions`.

        Returns:
            elbo_estimate: Estimate of the evidence lower bound based on samples from the
                variational posterior approximation.
        """
        log_prob = self.log_prob(parameters, *args, **kwargs)
        entropy = self.entropy(distributions)
        return log_prob + entropy

    def batch_elbo_estimate(self, size: Optional[th.Size] = None, *args, **kwargs) -> th.Tensor:
        """
        Evaluate an estimate of the evidence lower bound for a batch of parameter samples obtained
        from the variational approximation.

        Args:
            size: Batch size to evaluate evidence lower bound estimates for.
            distributions: Variational factors whose entropy to evaluate; defaults to
                :meth:`distributions`.

        Returns:
            elbo_estimate: Estimate of the evidence lower bound based on samples from the
                variational posterior approximation.
        """
        distributions = self.distributions()
        size = () if size is None else size
        parameters = self.rsample(size, distributions)
        elbo_estimate = self.elbo_estimate(parameters, *args, distributions=distributions, **kwargs)
        assert elbo_estimate.shape == size
        return elbo_estimate

    def check_log_prob_shape(self, shape=(2, 3), *args, **kwargs) -> None:
        """
        Check that :meth:`log_prob` returns the correct batch shape.
        """
        parameters = self.rsample(shape)
        log_prob = self.log_prob(parameters, *args, **kwargs)
        if log_prob.shape != shape:
            raise RuntimeError(f"expected batch shape {shape} but got {log_prob.shape}")


class TerminateOnPlateau:
    """
    Terminate training if a metric plateaus for a given number of steps.

    Args:
        patience: Number of steps to wait before terminating.
        max_num_steps: Maximum number of steps irrespective of termination criterion.
    """
    def __init__(self, patience: int, max_num_steps: Optional[int] = None) -> None:
        self.patience = patience
        self.best_value = float("inf")
        self.elapsed = 0
        self.num_steps = 0
        self.max_num_steps = max_num_steps or float("inf")

    def step(self, value: float) -> TerminateOnPlateau:
        """
        Update the state.

        Args:
            value: Value of the metric.

        Returns:
            instance: This instance.
        """
        if value < self.best_value:
            self.best_value = value
            self.elapsed = 0
        else:
            self.elapsed += 1
        self.num_steps += 1
        return self

    def __bool__(self) -> bool:
        """
        Return `True` if training should continue.
        """
        return self.elapsed < self.patience and self.num_steps < self.max_num_steps

    def __repr__(self) -> str:
        return f"TerminateOnPlateau(patience={self.patience}, " \
            f"max_num_steps={self.max_num_steps}, elapsed={self.elapsed}, " \
            f"num_steps={self.num_steps}, best_value={self.best_value})"


class optional(constraints.Constraint):
    """
    Validate a constraint if the value is not missing.
    """
    def __init__(self, constraint: constraints.Constraint) -> None:
        super().__init__()
        self.constraint = constraint
        self.event_dim = constraint.event_dim
        self.is_discrete = constraint.is_discrete

    def check(self, value):
        if value is None:
            return th.scalar_tensor(True)
        return self.constraint.check(value)


real_matrix = constraints.independent(constraints.real, 2)
