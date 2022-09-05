import numpy as np
from .missing_module import MissingModule
from .util import ArrayOrTensor, evaluate_squared_distance, is_tensor
try:
    import torch as th
except ModuleNotFoundError as ex:
    th = MissingModule(ex)


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
        is_tensor_ = is_tensor(x)
        exponent = - evaluate_squared_distance(x, y, self.period) / (2 * self.rho ** 2)
        cov = self.alpha * self.alpha * (exponent.exp() if is_tensor_ else np.exp(exponent))
        if self.epsilon:
            eye = th.eye(cov.shape[-1]) if is_tensor_ else np.eye(cov.shape[-1])
            return cov + self.epsilon * eye
        return cov
