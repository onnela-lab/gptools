from gptools.util.fft import transform_rfft, transform_irfft, evaluate_rfft_scale, \
    evaluate_rfft_log_abs_det_jacobian, transform_rfft2, transform_irfft2, evaluate_rfft2_scale, \
    evaluate_rfft2_log_abs_det_jacobian
import torch as th
from torch.distributions import constraints


real_matrix = constraints.independent(constraints.real, 2)


class FourierGaussianProcess1DTransform(th.distributions.Transform):
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        loc: Mean of the Gaussian process.
        cov: Covariance between a point of the origin with the rest of the space.
    """
    bijective = True
    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, loc: th.Tensor, cov: th.Tensor, cache_size: int = 0) -> None:
        super().__init__(cache_size)
        self.loc = loc
        self.cov = cov
        self._rfft_scale = evaluate_rfft_scale(self.cov)

    def _call(self, x: th.Tensor) -> th.Tensor:
        """
        Transform a Gaussian process realization to white noise.
        """
        return transform_rfft(x - self.loc, self.cov, self._rfft_scale)

    def _inv_call(self, y: th.Tensor) -> th.Tensor:
        """
        Transform white noise to a Gaussian process realization.
        """
        return transform_irfft(y, self.cov, self._rfft_scale) + self.loc

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return evaluate_rfft_log_abs_det_jacobian(self.cov, self._rfft_scale)


class FourierGaussianProcess1D(th.distributions.TransformedDistribution):
    """
    Fourier-based Gaussian process in one dimension.

    Args:
        loc: Mean of the Gaussian process.
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
        super().__init__(base_distribution, transform.inv, validate_args=validate_args)

    @property
    def loc(self):
        return self.transforms[0].inv.loc

    @property
    def cov(self):
        return self.transforms[0].inv.cov


class FourierGaussianProcess2DTransform(th.distributions.Transform):
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        loc: Mean of the Gaussian process.
        cov: Covariance between a point of the origin with the rest of the space.
    """
    bijective = True
    domain = real_matrix
    codomain = real_matrix

    def __init__(self, loc: th.Tensor, cov: th.Tensor, cache_size: int = 0) -> None:
        super().__init__(cache_size)
        self.loc = loc
        self.cov = cov
        self._rfft2_scale = evaluate_rfft2_scale(self.cov)

    def _call(self, x: th.Tensor) -> th.Tensor:
        """
        Transform a Gaussian process realization to white noise.
        """
        return transform_rfft2(x - self.loc, self.cov, self._rfft2_scale)

    def _inv_call(self, y: th.Tensor) -> th.Tensor:
        """
        Transform white noise to a Gaussian process realization.
        """
        return transform_irfft2(y, self.cov, self._rfft2_scale) + self.loc

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        return evaluate_rfft2_log_abs_det_jacobian(self.cov, self._rfft2_scale)


class FourierGaussianProcess2D(th.distributions.TransformedDistribution):
    """
    Fourier-based Gaussian process in two dimensions.

    Args:
        loc: Mean of the Gaussian process.
        cov: Covariance between a point of the origin with the rest of the space.
    """
    arg_constraints = {
        "loc": real_matrix,
        "cov": real_matrix,
    }
    support = real_matrix
    has_rsample = True

    def __init__(self, loc: th.Tensor, cov: th.Tensor, validate_args=None) -> None:
        *_, height, width = th.broadcast_shapes(loc.shape, cov.shape)
        shape = (height, width)
        base_distribution = th.distributions.Normal(th.zeros(shape), th.ones(shape))
        transform = FourierGaussianProcess2DTransform(loc, cov)
        super().__init__(base_distribution, transform.inv, validate_args=validate_args)

    @property
    def loc(self):
        return self.transforms[0].inv.loc

    @property
    def cov(self):
        return self.transforms[0].inv.cov
