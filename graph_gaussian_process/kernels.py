import numpy as np
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        cov = self.alpha ** 2 * np.exp(- evaluate_squared_distance(x) / (2 * self.rho ** 2))
        if self.epsilon:
            cov += self.epsilon * np.eye(cov.shape[-1])
        return cov
