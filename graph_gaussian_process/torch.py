import torch as th
from torch.distributions import constraints
import typing


DistributionDict = dict[str, th.distributions.Distribution]
TensorDict = dict[str, th.Tensor]


class GraphGaussianProcess(th.distributions.Distribution):
    """
    Gaussian process on a directec acyclic graph that is embedded in a metric space.

    Args:
        loc: Mean of the distribution.
        coords: Coordinates of nodes.
        predecessors: Matrix of node labels with shape `(num_nodes, max_degree)`. The row
            `predecessors[i]` denotes the predecessors of node `i` in the graph. Any unused slots
            should be set to -1. The first predecessor of each node must be itself, i.e., a self
            loop.
        kernel: Callable to evaluate the covariance which takes node coordinates as the only
            argument.
        lstsq_rcond: Threshold for rounding singular values to zero when solving the least squares
            problem for evaluating the parameters of conditional normal distributions.
        lstsq_driver: Method used to solve the least squares problem for evaluating the parameters
            of conditional normal distributions. Defaults to "gelsd" because the masked covariance
            matrices can be ill-conditioned, e.g., for nodes with degree much smaller than the
            maximum degree.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "coords": constraints.independent(constraints.real, 2),
        "predecessors": constraints.independent(constraints.real, 2),
    }
    support = constraints.real_vector
    has_rsample = False
    _ZERO = th.scalar_tensor(0, dtype=th.long)
    weights: th.Tensor

    def __init__(
            self, loc: th.Tensor, coords: th.Tensor, predecessors: th.LongTensor,
            kernel: typing.Callable[..., th.Tensor], lstsq_rcond: typing.Optional[float] = None,
            lstsq_driver: typing.Literal["gels", "gelsy", "gelsd", "gelss"] = "gelsd",
            validate_args=None) -> None:
        # Store parameter values and evaluate the shapes.
        self.num_nodes = loc.shape[-1]
        self.loc = th.as_tensor(loc)
        self.coords = th.as_tensor(coords)
        self.predecessors = th.as_tensor(predecessors)
        self.kernel = kernel
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]

        # Construct the indices (including the node itself), obtain the mask for valid parents, and
        # replace invalid parents with the first node. This could be any node because any associated
        # covariances will be masked out. But the first one does the trick.
        mask = self.predecessors >= 0
        self.indices = self.predecessors.maximum(self._ZERO)

        # Get all positions expanded to predecessors and evaluate the masked covariance function.
        cov = kernel(self.coords[..., self.indices, :]) * (mask[..., None, :] & mask[..., None])

        # Precompute intermediate values for evaluating the conditional parameters. The weights
        # correspond to the contributions of preceeding points to the location parameter.
        cov_iN = cov[..., 0, 1:]
        cov_NN = cov[..., 1:, 1:]
        self.weights, *_ = th.linalg.lstsq(cov_NN, cov_iN, rcond=lstsq_rcond,
                                           driver=lstsq_driver)
        self.scale = (cov[..., 0, 0] - (self.weights * cov_iN).sum(axis=-1)).sqrt()

        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        value = value - self.loc
        loc = (self.weights * value[..., self.indices[..., 1:]]).sum(axis=-1)
        return th.distributions.Normal(loc, self.scale).log_prob(value).sum(axis=-1)

    def sample(self, size: typing.Optional[th.Size] = None) -> th.Tensor:
        # We sample elements sequentially. This isn't particularly efficient due to the python loop
        # but it's much more efficient than inverting the whole matrix. TODO: look at sparse matrix
        # solvers that could solve this problem in a batch. First, sample white noise which we will
        # transform to a Gaussian process sample.
        ys = th.randn(th.Size(size or ()) + (self.num_nodes,))
        for i in range(self.num_nodes):
            loc = (self.weights[i] * ys[..., self.indices[i, 1:]]).sum(axis=-1)
            ys[..., i] = loc + self.scale[i] * ys[..., i]

        return ys + self.loc


class ParametrizedDistribution(th.nn.Module):
    """
    Parametrized distribution with initial conditions. Parameters are transformed to an
    unconstrained space for optimization, and the distribution is constructed by transforming back
    to the constrained space in the forward pass.

    Args:
        cls: Distribution to create.
        const: Sequence of parameter names that should be treated as constants, i.e., not require
            gradients.
        **kwargs: Initial conditions of the distribution.
    """
    def __init__(self, cls: type[th.distributions.Distribution], *,
                 const: typing.Optional[set[str]] = None, **kwargs: TensorDict):
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
    def __init__(self, approximations: dict[str, ParametrizedDistribution]) -> None:
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

    def rsample(self, size: typing.Optional[th.Size] = None,
                distributions: typing.Optional[DistributionDict] = None) -> TensorDict:
        """
        Draw reparametrized samples from the variational approximations.
        """
        distributions = distributions or self.distributions()
        return {key: distribution.rsample(size) for key, distribution in distributions.items()}

    def entropies(self, distributions: typing.Optional[DistributionDict] = None) -> TensorDict:
        """
        Get entropies of each variational approximation.
        """
        distributions = distributions or self.distributions()
        return {key: distribution.entropy() for key, distribution in distributions.items()}

    def entropy(self, distributions: typing.Optional[DistributionDict] = None) -> th.Tensor:
        """
        Get entropy of the joint variational approximation.

        Args:


        Returns:
            entropy: Entropy of the joint variational approximation as a scalar.
        """
        return sum(entropy.sum() for entropy in self.entropies(distributions).values())

    def elbo_estimate(
            self, parameters: TensorDict, *args,
            distributions: typing.Optional[DistributionDict] = None, **kwargs
            ) -> th.Tensor:
        """
        Evaluate an estimate of the evidence lower bound given parameter values.
        """
        log_prob = self.log_prob(parameters, *args, **kwargs)
        entropy = self.entropy(distributions)
        return log_prob + entropy

    def batch_elbo_estimate(self, size: typing.Optional[th.Size] = None, *args, **kwargs) \
            -> th.Tensor:
        """
        Evaluate an estimate of the evidence lower bound for a batch of parameter samples obtained
        from the variational approximation.
        """
        distributions = self.distributions()
        parameters = self.rsample(size)
        return self.elbo_estimate(parameters, *args, distributions=distributions, **kwargs)

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
    Terminate training if a metric plateaus for a given number of epochs.
    """
    def __init__(self, patience: int, max_num_steps: typing.Optional[int] = None) -> None:
        self.patience = patience
        self.best_value = float("inf")
        self.elapsed = 0
        self.num_steps = 0
        self.max_num_steps = max_num_steps or float("inf")

    def step(self, value: float):
        if value < self.best_value:
            self.best_value = value
            self.elapsed = 0
        else:
            self.elapsed += 1
        self.num_steps += 1
        return self

    def __bool__(self) -> bool:
        return self.elapsed < self.patience and self.num_steps < self.max_num_steps

    def __repr__(self) -> str:
        return f"TerminateOnPlateau(patience={self.patience}, " \
            f"max_num_steps={self.max_num_steps}, elapsed={self.elapsed}, " \
            f"num_steps={self.num_steps}, best_value={self.best_value})"
