from gptools.util.fft import transform_rfft, transform_irfft, evaluate_rfft_scale
import math
import torch as th
from torch.distributions import constraints


class FourierGaussianProcess1DTransform(th.distributions.Transform):
    """
    Transform white noise in the Fourier domain to a Gaussian process realization.

    Args:
        loc: Mean of the distribution.
        cov: Covariance between a point of the origin with the rest of the space.
    """
    bijective = True
    domain = constraints.real_vector
    codomain = constraints.real_vector
    _log2 = math.log(2)

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

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        *_, size = x.shape
        imagidx = (size + 1) // 2
        rfft_scale = evaluate_rfft_scale(self.cov)
        return rfft_scale.log().sum() + rfft_scale[1:imagidx].log().sum() \
            + self._log2 * ((size - 1) // 2) - size * math.log(size) / 2


class FourierGaussianProcess1D(th.distributions.TransformedDistribution):
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
        *_, size = th.broadcast_shapes(loc.shape, cov.shape)
        base_distribution = th.distributions.Normal(th.zeros(size), th.ones(size))
        transform = FourierGaussianProcess1DTransform(loc, cov)
        super().__init__(base_distribution, transform, validate_args=validate_args)

    @property
    def loc(self):
        return self.transforms[0].loc

    @property
    def cov(self):
        return self.transforms[0].cov
