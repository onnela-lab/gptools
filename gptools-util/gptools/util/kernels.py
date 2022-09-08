from . import ArrayOrTensor, ArrayOrTensorDispatch


dispatch = ArrayOrTensorDispatch()


def evaluate_residuals(x: ArrayOrTensor, y: ArrayOrTensor = None, period: ArrayOrTensor = None) \
        -> ArrayOrTensor:
    """
    Evaluate the residuals between points respecting periodic boundary conditions.

    If `period is not None` and boundary conditions apply, residuals have the correct "local"
    behavior, i.e., points to the left have a negative residual and points to the right have a
    positive residual. This leads to a discontinuity a distance `period / 2` from any reference
    point. The discontinuity is immaterial for even kernel functions.

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
    if period is None:
        return x - y
    residuals = (x % period) - (y % period)
    residuals = dispatch.where(residuals > period / 2, residuals - period, residuals)
    residuals = dispatch.where(residuals < - period / 2, residuals + period, residuals)
    return residuals


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
    residuals = evaluate_residuals(x, y, period)
    return (residuals * residuals).sum(axis=-1)


class Kernel:
    """
    Base class for covariance kernels.
    """
    def __init__(self, epsilon: float = 0, period: ArrayOrTensor = None):
        self.epsilon = epsilon
        self.period = period

    def __call__(self, x: ArrayOrTensor, y: ArrayOrTensor = None) -> ArrayOrTensor:
        cov = self._evaluate(x, y)
        if self.epsilon:
            return cov + self.epsilon * dispatch[cov].eye(cov.shape[-1])
        return cov

    def _evaluate(self, x: ArrayOrTensor, y: ArrayOrTensor = None) -> ArrayOrTensor:
        raise NotImplementedError

    @property
    def is_periodic(self):
        return self.period is not None


class ExpQuadKernel(Kernel):
    r"""
    Exponentiated quadratic kernel.

    .. math::

        \text{cov}\left(x, y\right) = \alpha^2 \exp\left(-\frac{\left(x-y\right)^2}{2\rho^2}\right)

    Args:
        alpha: Scale of the covariance.
        rho: Correlation length.
        epsilon: Additional diagonal variance.
        period: Period for circular boundary conditions.
    """
    def __init__(self, alpha: float, rho: float, epsilon: float = 0, period: ArrayOrTensor = None) \
            -> None:
        super().__init__(epsilon, period)
        self.alpha = alpha
        self.rho = rho

    def _evaluate(self, x: ArrayOrTensor, y: ArrayOrTensor = None) -> ArrayOrTensor:
        exponent = - evaluate_squared_distance(x, y, self.period) / (2 * self.rho ** 2)
        return self.alpha * self.alpha * dispatch.exp(exponent)
