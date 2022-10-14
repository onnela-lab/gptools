import math
import operator
from typing import Callable, Optional
from . import ArrayOrTensor, ArrayOrTensorDispatch, OptionalArrayOrTensor


dispatch = ArrayOrTensorDispatch()


def evaluate_residuals(x: ArrayOrTensor, y: OptionalArrayOrTensor = None,
                       period: OptionalArrayOrTensor = None) -> ArrayOrTensor:
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

    Example:
      .. plot::

        from gptools.util.kernels import evaluate_residuals
        from matplotlib import pyplot as plt
        import numpy as np

        width = 3  # Width of the domain.
        step_seq = [6, 7]  # Number of grid points in the domain.
        boundary_color = "silver"  # Color for domain boundaries.

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

        for ax, steps in zip(axes, step_seq):
            # Plot distances for two locations.
            x = np.linspace(-width, 2 * width, 3 * steps + 1, endpoint=True)
            ys = [x[steps + 2], x[2 * steps - 1]]
            for y in ys:
                line, = ax.plot(x, evaluate_residuals(x, y, width), marker=".")
                ax.scatter(y, 0, color=line.get_color(), label=f"$y={y:.1f}$")

            # Plot boundary indicators.
            for i in range(-1, 3):
                ax.axvline(i * width, color=boundary_color, ls="--")
            ax.axhline(width / 2, color=boundary_color, ls="--")
            ax.axhline(-width / 2, color=boundary_color, ls="--", label="domain boundaries")
            ax.axhline(0, color="silver", ls=":")

            ax.set_aspect("equal")
            ax.set_ylabel(r"residual $x - y$")
            ax.set_title(fr"$n={steps}$")

        axes[1].set_xlabel("position $x$")

        # Adjust boundaries.
        factor = 0.2
        ax.set_ylim(-(1 + factor) * width / 2, (1 + factor) * width / 2)
        ax.set_xlim(-(1 + factor / 2) * width, (2 + factor / 2) * width)
        fig.tight_layout()
    """
    # Expand the shape so we get the Cartesian product of elements in x (while keeping the batch
    # shape).
    if y is None:
        x, y = x[..., :, None, :], x[..., None, :, :]
    residuals = x - y
    if period is not None:
        residuals = residuals - period * dispatch.round(residuals / period)
    return residuals


def evaluate_squared_distance(x: ArrayOrTensor, y: OptionalArrayOrTensor = None,
                              period: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    r"""
    Evaluate the squared distance between points respecting periodic boundary conditions.

    Args:
        x: Coordinates with shape `(..., p)`, where `...` is the batch shape and `p` is the number
            of dimensions of the embedding space.
        y: Coordinates with shape `(..., p)` which must be broadcastable to `x`. If not given, the
            distance between the Cartesian product of `x` will be evaluated.
        period: Period of circular boundary conditions.

    Returns:
        dist2: Squared distance between `x` and `y`.

    Example:
      .. plot::

        from gptools.util import coordgrid
        from gptools.util.kernels import evaluate_squared_distance
        from matplotlib import pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        height = 40
        width = 50
        shape = (height, width)
        xs = coordgrid(np.arange(height), np.arange(width))

        idx = len(xs) // 4 + width // 4
        periods = [None, shape]
        dists = [np.sqrt(evaluate_squared_distance(xs[idx], xs, period=period))
                for period in periods]

        vmax = np.sqrt(width ** 2 + height ** 2)
        for ax, period, dist in zip(axes, periods, dists):
            dist = dist.reshape(shape)
            im = ax.imshow(dist, vmax=vmax, origin="lower")
            colorbar = fig.colorbar(im, ax=ax, location="top")
            label = r"distance $d\left(x,y\right)$"
            if period:
                label += " (periodic boundaries)"
            colorbar.set_label(label)
            cs = ax.contour(dist, colors="w", levels=[10, 20, 30], linestyles=["-", "--", ":"])
            plt.clabel(cs)
            ax.scatter(*np.unravel_index(idx, shape)[::-1], color="C1").set_edgecolor("w")
            ax.set_xlabel("position $x$")

        axes[0].set_ylabel("position $y$")
        fig.tight_layout()
    """
    residuals = evaluate_residuals(x, y, period)
    return (residuals * residuals).sum(axis=-1)


class Kernel:
    """
    Base class for covariance kernels.

    Args:
        period: Period for circular boundary conditions.
    """
    def __init__(self, period: OptionalArrayOrTensor = None):
        self.period = period

    def __call__(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        return self.evaluate(x, y)

    def evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        """
        Evaluate the covariance kernel.

        Args:
            x: First set of points.
            y: Second set of points (defaults to `x` for pairwise covariances).

        Returns:
            cov: Covariance between the two sets of points.
        """
        raise NotImplementedError

    def __add__(self, other) -> "CompositeKernel":
        return CompositeKernel(operator.add, self, other)

    def __mul__(self, other) -> "CompositeKernel":
        return CompositeKernel(operator.mul, self, other)

    @property
    def is_periodic(self):
        return self.period is not None


class CompositeKernel(Kernel):
    """
    Composition of two kernels.

    Args:
        operation: Operation for composing kernels.
        a: First kernel.
        b: Second kernel.
    """
    def __init__(self, operation: Callable, a: Kernel, b: Kernel) -> None:
        period = None
        if isinstance(a, Kernel) and isinstance(b, Kernel):
            if a.is_periodic != b.is_periodic:
                raise ValueError("either both or neither kernel must be periodic")
            if a.is_periodic:
                if not dispatch.allclose(a.period, b.period):
                    raise ValueError("kernels do not have the same period")
                period = a.period
        super().__init__(period)
        self.operation = operation
        self.a = a
        self.b = b

    def evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        return self.operation(self.a(x, y) if callable(self.a) else self.a,
                              self.b(x, y) if callable(self.b) else self.b)


class DiagonalKernel(Kernel):
    """
    Diagonal kernel with "nugget" variance. The kernel can only evaluated pairwise for a single set
    of points but not for the Cartesian product of two sets of points.
    """
    def __init__(self, epsilon: float = 1, period: OptionalArrayOrTensor = None) -> None:
        super().__init__(period)
        self.epsilon = epsilon

    def evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        if y is not None:
            raise ValueError
        return dispatch[x].eye(x.shape[-2]) * self.epsilon


class ExpQuadKernel(Kernel):
    r"""
    Exponentiated quadratic kernel.

    .. math::

        \text{cov}\left(x, y\right) = \sigma^2 \exp\left(-\frac{\left(x-y\right)^2}{2\ell^2}\right)

    Args:
        sigma: Scale of the covariance.
        length_scale: Correlation length.
        period: Period for circular boundary conditions.
    """
    def __init__(self, sigma: float, length_scale: float, period: OptionalArrayOrTensor = None) \
            -> None:
        super().__init__(period)
        self.sigma = sigma
        self.length_scale = length_scale

    def evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        residuals = evaluate_residuals(x, y, self.period) / self.length_scale
        exponent = - dispatch.square(residuals).sum(axis=-1) / 2
        return self.sigma * self.sigma * dispatch.exp(exponent)


class HeatKernel(Kernel):
    """
    Heat kernel on a finite domain with periodic boundary conditions.

    Args:
        sigma: Scale of the covariance.
        length_scale: Correlation length.
        period: Period for circular boundary conditions.
        num_terms: Number of terms in the series approximation of the heat equation solution.
    """
    def __init__(self, sigma: ArrayOrTensor, length_scale: ArrayOrTensor, period: ArrayOrTensor,
                 num_terms: Optional[int] = None) -> None:
        super().__init__(period)
        self.sigma = sigma
        self.length_scale = length_scale
        # Evaluate the effective relaxation time of the heat kernel.
        self.time = 2 * math.pi ** 2 * (self.length_scale / self.period) ** 2
        # The terms decay rapidly with exp(- k^2 * time) so we only need to consider the first few.
        # If not given, we try to reach k^2 * time > 10.
        self.num_terms = num_terms or int((10 / self.time) ** 2) + 1

    def evaluate(self, x, y=None):
        # TODO: make this work for higher dimensions.
        residuals = evaluate_residuals(x, y, self.period).squeeze() / self.period
        ks = dispatch[x].arange(1, self.num_terms)
        parts = dispatch.cos(2 * math.pi * ks * residuals[..., None]) \
            * dispatch.exp(- ks ** 2 * self.time)
        value = 2 * parts.sum(axis=-1) + 1
        return self.sigma ** 2 * value * dispatch.sqrt(self.time / math.pi)

    def evaluate_rfft(self, shape: tuple[int]) -> ArrayOrTensor:
        """
        Evaluate the real fast Fourier transform of the kernel.

        Args:
            shape: Number of sample points in each dimension.

        Returns:
            rfft: Fourier coefficients with shape `(*shape[:-1], shape[-1] // 2 + 1)`.
        """
        import numpy as np
        size = math.prod(shape)
        ks = np.arange(size // 2 + 1)
        return size * math.sqrt(2 * math.pi) * self.length_scale * self.sigma ** 2 / self.period \
            * dispatch.exp(-2 * (math.pi * ks * self.length_scale / self.period) ** 2)
