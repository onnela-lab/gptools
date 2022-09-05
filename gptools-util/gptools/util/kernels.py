from . import ArrayOrTensor, ArrayOrTensorDispatch


dispatch = ArrayOrTensorDispatch()


def evaluate_squared_distance(x: ArrayOrTensor, y: ArrayOrTensor = None,
                              period: ArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the squared distance between points respecting periodic boundary conditions.

    Args:
        x: Coordinates with shape `(..., p)`, where `...` is the batch shape and `p` is the number
            of dimensions of the embedding space.
        y: Coordinates with shape `(..., p)` which must be broadcastable to `x`. If not given, the
            distance between the Cartesian product of `x` will be evaluated.
        period: Period of circular boundary conditions.

    Returns:
        dist2: Squared distance between `x` and `y`.
    """
    # Expand the shape so we get the Cartesian product of elements in x (while keeping the batch
    # shape).
    if y is None:
        x, y = x[..., :, None, :], x[..., None, :, :]
    residuals: ArrayOrTensor = x - y
    if period is not None:
        residuals = residuals % period
        residuals = dispatch.minimum(residuals, period - residuals)
    return (residuals * residuals).sum(axis=-1)


class ExpQuadKernel:
    """
    Exponentiated quadratic kernel.

    Args:
        alpha: Scale of the covariance.
        rho: Correlation length.
        epsilon: Additional diagonal variance.
        period: Period for circular boundary conditions.
    """
    def __init__(self, alpha: float, rho: float, epsilon: float = 0, period: ArrayOrTensor = None) \
            -> None:
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon
        self.period = period

    def __call__(self, x: ArrayOrTensor, y: ArrayOrTensor = None) -> ArrayOrTensor:
        exponent = - evaluate_squared_distance(x, y, self.period) / (2 * self.rho ** 2)
        cov = self.alpha * self.alpha * dispatch.exp(exponent)
        if self.epsilon:
            return cov + self.epsilon * dispatch[x].eye(cov.shape[-1])
        return cov
