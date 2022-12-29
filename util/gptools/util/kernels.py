import math
import numpy as np
import operator
from typing import Callable, Iterable, Union
from . import ArrayOrTensor, ArrayOrTensorDispatch, coordgrid, OptionalArrayOrTensor
from .fft import expand_rfft


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

    def evaluate_rfft(self, shape: Union[int, tuple[int]]) -> ArrayOrTensor:
        """
        Evaluate the real fast Fourier transform of the kernel.

        Args:
            shape: Number of sample points in each dimension.

        Returns:
            rfft: Fourier coefficients with shape `(*shape[:-1], shape[-1] // 2 + 1)`.
        """
        raise NotImplementedError

    def __add__(self, other) -> "CompositeKernel":
        return CompositeKernel(operator.add, self, other)

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
        return self.operation(self.a.evaluate(x, y) if isinstance(self.a, Kernel) else self.a,
                              self.b.evaluate(x, y) if isinstance(self.b, Kernel) else self.b)


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
    Exponentiated quadratic kernel or solution to the heat equation if the kernel is periodic.

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

    def evaluate_rfft(self, shape: Union[int, tuple[int]]) -> ArrayOrTensor:
        if not self.is_periodic:
            raise NotImplementedError
        if not isinstance(shape, Iterable):
            shape = [shape]
        ndim = len(shape)
        rescaled_length_scale = self.length_scale * np.ones(ndim) / self.period
        value = None
        for i, size in enumerate(shape):
            xi = np.arange(size // 2 + 1)
            part = size * rescaled_length_scale[i] * (2 * math.pi) ** 0.5 \
                * np.exp(-2 * (math.pi * xi * rescaled_length_scale[i]) ** 2)
            if i != ndim - 1:
                part = expand_rfft(part, size)
            if value is None:
                value = part
            else:
                value = value[..., None] * part

        return self.sigma ** 2 * value


class MaternKernel(Kernel):
    """
    Matern covariance function.

    Args:
        dof: Smoothness parameter.
        sigma: Scale of the covariance.
        length_scale: Correlation length.
        period: Period for circular boundary conditions.
    """
    def __init__(self, dof: float, sigma: float, length_scale: float,
                 period: OptionalArrayOrTensor = None) -> None:
        super().__init__(period)
        if dof not in (allowed_dofs := {3 / 2, 5 / 2}):
            raise ValueError(f"dof must be one of {allowed_dofs} but got {dof}")
        self.sigma = sigma
        self.length_scale = length_scale
        self.dof = dof

    def evaluate(self, x: ArrayOrTensor, y: OptionalArrayOrTensor = None) -> ArrayOrTensor:
        residuals = evaluate_residuals(x, y, self.period) / self.length_scale
        distance = (2 * self.dof * residuals * residuals).sum(axis=-1) ** 0.5
        if self.dof == 3 / 2:
            value = 1 + distance
        elif self.dof == 5 / 2:
            value = 1 + distance + distance * distance / 3
        else:
            raise NotImplementedError
        return self.sigma * self.sigma * value * dispatch.exp(-distance)

    def evaluate_rfft(self, shape: tuple[int]):
        if not self.is_periodic:
            raise NotImplementedError
        if not isinstance(shape, Iterable):
            shape = [shape]
        from scipy import special

        # Construct the grid to evaluate on.
        size = np.prod(shape)
        *head, tail = shape
        ks = [np.arange(n) for n in head]
        ks.append(np.arange(tail // 2 + 1))
        ks = coordgrid(*ks)
        ks = np.minimum(ks, shape - ks)
        ndim = len(shape)

        # Evaluate the spectral density.
        length_scale = self.length_scale / self.period * np.ones(ndim)
        arg = 1 + 2 / self.dof * np.sum((math.pi * length_scale * ks) ** 2, axis=-1)
        value = size * 2 ** ndim * (math.pi / (2 * self.dof)) ** (ndim / 2) \
            * special.gamma(self.dof + ndim / 2) / special.gamma(self.dof) \
            * arg ** -(self.dof + ndim / 2) * length_scale.prod()
        return self.sigma * self.sigma * value.reshape(head + [tail // 2 + 1])
