import torch as th
from torch.distributions import constraints
from typing import Callable, Literal, Optional


class GraphGaussianProcess(th.distributions.Distribution):
    """
    Gaussian process on a directec acyclic graph that is embedded in a metric space.

    Args:
        loc: Mean of the distribution.
        coords: Coordinates of nodes.
        predecessors: Matrix of node labels with shape `(num_nodes, max_degree)`. The row
            `predecessors[i]` denotes the predecessors of node `i` in the graph. Any unused slots
            should be set to -1.
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
            kernel: Callable[..., th.Tensor], lstsq_rcond: Optional[float] = None,
            lstsq_driver: Literal["gels", "gelsy", "gelsd", "gelss"] = "gelsd",
            validate_args=None) -> None:
        # Store parameter values and evaluate the shapes.
        self.num_nodes = loc.shape[-1]
        self.loc = th.as_tensor(loc)
        self.coords = th.as_tensor(coords)
        # Prefix the self-loops for the predecessors.
        self.predecessors = th.hstack([th.arange(self.num_nodes)[:, None],
                                       th.as_tensor(predecessors)])
        self.kernel = kernel
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]

        # Construct the indices (including the node itself), obtain the mask for valid parents, and
        # replace invalid parents with the first node. This could be any node because any associated
        # covariances will be masked out. But the first one does the trick.
        mask = self.predecessors >= 0
        self.indices = self.predecessors.maximum(self._ZERO)

        # Get all positions expanded to predecessors and evaluate the masked covariance function.
        cov = kernel.evaluate(self.coords[..., self.indices, :]) \
            * (mask[..., None, :] & mask[..., None])

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

    def sample(self, size: Optional[th.Size] = None) -> th.Tensor:
        # We sample elements sequentially. This isn't particularly efficient due to the python loop
        # but it's much more efficient than inverting the whole matrix. TODO: look at sparse matrix
        # solvers that could solve this problem in a batch. First, sample white noise which we will
        # transform to a Gaussian process sample.
        ys = th.randn(th.Size(size or ()) + (self.num_nodes,))
        for i in range(self.num_nodes):
            loc = (self.weights[i] * ys[..., self.indices[i, 1:]]).sum(axis=-1)
            ys[..., i] = loc + self.scale[i] * ys[..., i]

        return ys + self.loc
