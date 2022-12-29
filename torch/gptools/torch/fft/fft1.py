from gptools.util.fft import transform_rfft, transform_irfft, evaluate_rfft_log_abs_det_jacobian
from gptools.util.fft.fft1 import _get_rfft_scale
import torch as th
from torch.distributions import constraints
from .. import OptionalTensor
from ..util import optional


class FourierGaussianProcess1DTransform(th.distributions.Transform):
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.
    """
    bijective = True
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, loc: th.Tensor, *, cov_rfft: OptionalTensor = None,
                 cov: OptionalTensor = None, rfft_scale: OptionalTensor = None,
                 cache_size: int = 0) -> None:
        super().__init__(cache_size)
        self.loc = loc
        self.cov_rfft = cov_rfft
        self.cov = cov
        self.rfft_scale = _get_rfft_scale(self.cov_rfft, self.cov, rfft_scale, self.loc.shape[-1])

    def _call(self, x: th.Tensor) -> th.Tensor:
        """
        Transform a Gaussian process realization to white noise.
        """
        return transform_rfft(x, self.loc, rfft_scale=self.rfft_scale)

    def _inv_call(self, y: th.Tensor) -> th.Tensor:
        """
        Transform white noise to a Gaussian process realization.
        """
        return transform_irfft(y, self.loc, rfft_scale=self.rfft_scale)

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return evaluate_rfft_log_abs_det_jacobian(x.shape[-1], rfft_scale=self.rfft_scale)


class FourierGaussianProcess1D(th.distributions.TransformedDistribution):
    """
    Fourier-based Gaussian process in one dimension.

    Args:
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.
    """
    arg_constraints = {
        "loc": constraints.real_vector,
        "cov_rfft": optional(constraints.real_vector),
        "cov": optional(constraints.real_vector),
        "rfft_scale": constraints.real_vector,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc: th.Tensor, *, cov_rfft: OptionalTensor = None, cov:
                 OptionalTensor = None, rfft_scale: OptionalTensor = None, validate_args=None) \
            -> None:
        *_, size = loc.shape
        base_distribution = th.distributions.Normal(th.zeros(size), th.ones(size))
        transform = FourierGaussianProcess1DTransform(loc, cov_rfft=cov_rfft, cov=cov,
                                                      rfft_scale=rfft_scale)
        super().__init__(base_distribution, transform.inv, validate_args=validate_args)

    @property
    def loc(self):
        return self.transforms[0].inv.loc

    @property
    def cov_rfft(self):
        return self.transforms[0].inv.cov_rfft

    @property
    def cov(self):
        return self.transforms[0].inv.cov

    @property
    def rfft_scale(self):
        return self.transforms[0].inv.rfft_scale
