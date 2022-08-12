import torch as th
from torch.distributions import constraints
import typing


class GraphGaussianProcess(th.distributions.Distribution):
    """
    Gaussian process on a directec acyclic graph that is embedded in a metric space.

    Args:
        loc: Mean of the distribution.
        coords: Coordinates of nodes.
        neighborhoods: Matrix of node labels with shape `(num_nodes, max_degree)`. The row
            `neighborhoods[i]` denotes the predecessors of node `i` in the graph. Any unused slots
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
        "neighborhoods": constraints.independent(constraints.real, 2),
    }
    support = constraints.real_vector
    has_rsample = False
    _ZERO = th.scalar_tensor(0, dtype=th.long)
    weights: th.Tensor

    def __init__(
            self, loc: th.Tensor, coords: th.Tensor, neighborhoods: th.LongTensor,
            kernel: typing.Callable[..., th.Tensor], lstsq_rcond: typing.Optional[float] = None,
            lstsq_driver: typing.Literal["gels", "gelsy", "gelsd", "gelss"] = "gelsd",
            validate_args=None) -> None:
        # Store parameter values and evaluate the shapes.
        self.num_nodes = loc.shape[-1]
        self.loc = loc
        self.coords = coords
        self.neighborhoods = neighborhoods
        self.kernel = kernel
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]

        # Construct the indices (including the node itself), obtain the mask for valid parents, and
        # replace invalid parents with the first node. This could be any node because any associated
        # covariances will be masked out. But the first one does the trick.
        mask = self.neighborhoods >= 0
        self.indices = self.neighborhoods.maximum(self._ZERO)

        # Get all positions expanded to neighborhoods and evaluate the masked covariance function.
        cov = kernel(coords[..., self.indices, :]) * (mask[..., None, :] & mask[..., None])

        # Precompute intermediate values for evaluating the conditional parameters. The weights
        # correspond to the contributions of neighboring points to the location parameter.
        self.cov_ii = cov[..., 0, 0]
        self.cov_iN = cov[..., 0, 1:]
        cov_NN = cov[..., 1:, 1:]
        self.weights, *_ = th.linalg.lstsq(cov_NN, self.cov_iN, rcond=lstsq_rcond,
                                           driver=lstsq_driver)

        super().__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        value = value - self.loc
        loc = (self.weights * value[..., self.indices[..., 1:]]).sum(axis=-1)
        scale = (self.cov_ii - (self.weights * self.cov_iN).sum(axis=-1)).sqrt()
        return th.distributions.Normal(loc, scale).log_prob(value).sum(axis=-1)

    def sample(self, size: th.Size) -> th.Tensor:
        raise NotImplementedError
