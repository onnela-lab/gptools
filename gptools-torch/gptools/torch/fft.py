from gptools.util.fft import evaluate_log_prob_rfft, transform_rfft, transform_irfft
import torch as th
from torch.distributions import constraints
import typing


class FourierGaussianProcess1D(th.distributions.Distribution):
    """
    Fourier-based Gaussian process in one dimension.

    Args:
        loc: Mean of the distribution.
        cov: Covariance between a point of the origin with the rest of the space.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov": constraints.real_vector,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc: th.Tensor, cov: th.Tensor, validate_args=None) -> None:
        *batch_shape, size = th.broadcast_shapes(loc.shape, cov.shape)
        self.loc = loc
        self.cov = cov
        super().__init__(batch_shape, (size,), validate_args)

    def log_prob(self, value: th.Tensor) -> th.Tensor:
        return evaluate_log_prob_rfft(value - self.loc, self.cov)

    def rsample(self, sample_shape: typing.Optional[th.Size] = None):
        raise NotImplementedError


class FourierGaussianProcess1DTransform(th.distributions.Transform):
    """
    Transform white noise to a Gaussian process realization.

    Args:
        loc: Mean of the distribution.
        cov: Covariance between a point of the origin with the rest of the space.
    """
    bijective = True
    domain: constraints.real_vector
    codomain: constraints.real_vector

    def __init__(self, loc: th.Tensor, cov: th.Tensor, cache_size: int = 0) -> None:
        super().__init__(cache_size)
        self.loc = loc
        self.cov = cov

    def _call(self, x: th.Tensor) -> th.Tensor:
        """
        Transform white noise to a Gaussian process realization.
        """
        return transform_rfft(x, self.cov) + self.loc

    def _inv_call(self, y: th.Tensor) -> th.Tensor:
        """
        Transform a Gaussian process realization to white noise.
        """
        return transform_irfft(y - self.loc, self.cov)
