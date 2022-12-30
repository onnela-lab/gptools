from gptools.util.fft import transform_rfft2, transform_irfft2, evaluate_rfft2_log_abs_det_jacobian
from gptools.util.fft.fft2 import _get_rfft2_scale
import torch as th
from .. import OptionalTensor
from ..util import real_matrix, optional


class FourierGaussianProcess2DTransform(th.distributions.Transform):
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        loc: Mean of the Gaussian process with shape `(..., height, width)`.
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft2_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.
    """
    bijective = True
    domain = real_matrix
    codomain = real_matrix

    def __init__(self, loc: th.Tensor, *, cov_rfft2: OptionalTensor = None,
                 cov: OptionalTensor = None, rfft2_scale: OptionalTensor = None,
                 cache_size: int = 0) -> None:
        super().__init__(cache_size)
        self.loc = loc
        self.cov_rfft2 = cov_rfft2
        self.cov = cov
        self.rfft2_scale = _get_rfft2_scale(self.cov_rfft2, self.cov, rfft2_scale,
                                            self.loc.shape[-1])

    def _call(self, x: th.Tensor) -> th.Tensor:
        """
        Transform a Gaussian process realization to white noise.
        """
        return transform_rfft2(x, self.loc, rfft2_scale=self.rfft2_scale)

    def _inv_call(self, y: th.Tensor) -> th.Tensor:
        """
        Transform white noise to a Gaussian process realization.
        """
        return transform_irfft2(y, self.loc, rfft2_scale=self.rfft2_scale)

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return evaluate_rfft2_log_abs_det_jacobian(x.shape[-1], rfft2_scale=self.rfft2_scale)


class FourierGaussianProcess2D(th.distributions.TransformedDistribution):
    """
    Fourier-based Gaussian process in two dimensions.

    Args:
        loc: Mean of the Gaussian process with shape `(..., height, width)`.
        cov_rfft2: Precomputed real fast Fourier transform of the kernel with shape
            `(..., height, width // 2 + 1)`.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., height, width)`.
        rfft2_scale: Optional precomputed scale of Fourier coefficients with shape
            `(..., height, width // 2 + 1)`.
    """
    arg_constraints = {
        "loc": real_matrix,
        "cov_rfft2": optional(real_matrix),
        "cov": optional(real_matrix),
        "rfft2_scale": real_matrix,
    }
    support = real_matrix
    has_rsample = True

    def __init__(self, loc: th.Tensor, *, cov_rfft2: OptionalTensor = None,
                 cov: OptionalTensor = None, rfft2_scale: OptionalTensor = None,
                 validate_args=None) -> None:
        *_, height, width = loc.shape
        shape = (height, width)
        base_distribution = th.distributions.Normal(th.zeros(shape), th.ones(shape))
        transform = FourierGaussianProcess2DTransform(loc, cov_rfft2=cov_rfft2, cov=cov,
                                                      rfft2_scale=rfft2_scale)
        super().__init__(base_distribution, transform.inv, validate_args=validate_args)

    @property
    def loc(self):
        return self.transforms[0].inv.loc

    @property
    def cov_rfft2(self):
        return self.transforms[0].inv.cov_rfft2

    @property
    def cov(self):
        return self.transforms[0].inv.cov

    @property
    def rfft2_scale(self):
        return self.transforms[0].inv.rfft2_scale
