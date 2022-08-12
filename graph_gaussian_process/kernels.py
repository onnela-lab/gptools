import numpy as np
import torch as th
import typing
from .util import evaluate_squared_distance


class ExpQuadKernel:
    """
    Exponentiated quadratic kernel.

    Args:
        alpha: Scale of the covariance.
        rho: Correlation length.
    """
    def __init__(self, alpha: float, rho: float, epsilon: float = 0) -> None:
        self.alpha = alpha
        self.rho = rho
        self.epsilon = epsilon

    def __call__(self, x: typing.Union[np.ndarray, th.Tensor]) \
            -> typing.Union[np.ndarray, th.Tensor]:
        is_tensor = isinstance(x, th.Tensor)
        exponent = - evaluate_squared_distance(x) / (2 * self.rho ** 2)
        cov = self.alpha * self.alpha * (exponent.exp() if is_tensor else np.exp(exponent))
        if self.epsilon:
            eye = th.eye(cov.shape[-1]) if is_tensor else np.eye(cov.shape[-1])
            return cov + self.epsilon * eye
        return cov
