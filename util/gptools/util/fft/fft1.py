import math
import numpy as np
from typing import Optional
from .. import mutually_exclusive_kwargs
from .util import log2, log_prob_stdnorm, sqrt2


def _get_rfft_scale(cov_rfft: Optional[np.ndarray], cov: Optional[np.ndarray],
                    rfft_scale: Optional[np.ndarray], size: Optional[int]) -> np.ndarray:
    num_given = sum([cov_rfft is not None, cov is not None, rfft_scale is not None])
    if num_given != 1:  # pragma: no cover
        raise ValueError("exactly one of `cov_rfft`, `cov`, or `rfft_scale` must be given")
    if rfft_scale is not None:
        return rfft_scale
    return evaluate_rfft_scale(cov_rfft=cov_rfft, cov=cov, size=size)


@mutually_exclusive_kwargs("cov", "cov_rfft")
def evaluate_rfft_scale(*, cov: Optional[np.ndarray] = None,
                        cov_rfft: Optional[np.ndarray] = None, size: Optional[int] = None) \
        -> np.ndarray:
    """
    Evaluate the scale of Fourier coefficients.

    Args:
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            :code:`(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape :code:`(..., size)`.
        size: Size of the real signal. Necessary because the size cannot be inferred from
            :code:`rfft`.

    Returns:
        scale: Scale of Fourier coefficients with shape :code:`(..., size // 2 + 1)`.
    """
    if cov_rfft is None:
        *_, size = cov.shape
        cov_rfft = np.fft.rfft(cov).real
    scale: np.ndarray = np.sqrt(size * cov_rfft / 2)
    # Rescale for the real-only zero frequency term.
    scale[0] *= sqrt2
    if size % 2 == 0:
        # Rescale for the real-only Nyqvist frequency term.
        scale[..., -1] *= sqrt2
    return scale


def expand_rfft(rfft: np.ndarray, n: int) -> np.ndarray:
    """
    Convert truncated real Fourier coefficients to full Fourier coefficients.

    Args:
        rfft: Truncated real Fourier coefficients with shape :code:`(n // 2 + 1,)`.
        n: Number of samples.

    Returns:
        fft: Full Fourier coefficients with shape :code:`(n,)`.
    """
    nrfft = n // 2 + 1
    ncomplex = (n - 1) // 2
    fft = np.empty(n, dtype=rfft.dtype)
    fft[:nrfft] = rfft
    fft[nrfft:] = np.flip(rfft[1:1 + ncomplex]).conj()
    return fft


def unpack_rfft(z: np.ndarray, size: int) -> np.ndarray:
    """
    Unpack the Fourier coefficients of a real Fourier transform with :code:`size // 2 + 1` elements
    to a vector of :code:`size` elements.

    Args:
        z: Real Fourier transform coefficients.
        size: Size of the real signal. Necessary because the size cannot be inferred from
            :code:`rfft`.

    Returns:
        z: Unpacked vector of :code:`size` elements comprising the :code:`size // 2 + 1` real parts
            of the zero frequency term, complex terms, and Nyqvist frequency term (for even
            :code:`size`). The subsequent :code:`(size - 1) // 2` elements are the imaginary parts
            of complex coefficients.
    """
    ncomplex = (size - 1) // 2
    parts = [z.real, z.imag[..., 1: ncomplex + 1]]
    return np.concatenate(parts, axis=-1)


def pack_rfft(z: np.ndarray, full_fft: bool = False) -> np.ndarray:
    """
    Transform a real vector with :code:`size` elements to a vector of complex Fourier coefficients
    with :code:`size // 2 + 1` elements ready for inverse real fast Fourier transformation.

    Args:
        z: Unpacked vector of :code:`size` elements. See :func:`unpack_rfft` for details.
        full_fft: Whether to return the full set of Fourier coefficients rather than just the
            reduced representation for the real fast Fourier transform. The full representation is
            required for :func:`pack_rfft2`.

    Returns:
        rfft: Real Fourier transform coefficients.
    """
    *_, size = z.shape
    fftsize = size // 2 + 1
    ncomplex = (size - 1) // 2
    # Zero frequency term, real parts of complex coefficients and possible Nyqvist frequency.
    rfft = z[..., :fftsize] * (1 + 0j)
    # Imaginary parts of complex coefficients.
    rfft[..., 1:ncomplex + 1] += 1j * z[..., fftsize:]
    if not full_fft:
        return rfft
    # Add the redundant complex coefficients.
    return np.concatenate([rfft, np.flip(rfft[..., 1:ncomplex + 1].conj(), (-1,))], -1)


def transform_irfft(z: np.ndarray, loc: np.ndarray, *, cov_rfft: Optional[np.ndarray] = None,
                    cov: Optional[np.ndarray] = None, rfft_scale: Optional[np.ndarray] = None) \
        -> np.ndarray:
    """
    Transform white noise in the Fourier domain to a Gaussian process realization.

    Args:
        z: Fourier-domain white noise with shape :code:`(..., size)`. See :func:`unpack_rfft` for
            details.
        loc: Mean of the Gaussian process with shape :code:`(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            :code:`(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape :code:`(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape
            :code:`(..., size // 2 + 1)`.

    Returns:
        y: Realization of the Gaussian process with shape :code:`(..., size)`.
    """
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, z.shape[-1])
    rfft = pack_rfft(z) * rfft_scale
    return np.fft.irfft(rfft, z.shape[-1]) + loc


def transform_rfft(y: np.ndarray, loc: np.ndarray, *, cov_rfft: Optional[np.ndarray] = None,
                   cov: Optional[np.ndarray] = None, rfft_scale: Optional[np.ndarray] = None) \
        -> np.ndarray:
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        y: Realization of the Gaussian process with shape :code:`(..., size)`.
        loc: Mean of the Gaussian process with shape :code:`(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            :code:`(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape :code:`(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape
            :code:`(..., size // 2 + 1)`.

    Returns:
        z: Fourier-domain white noise with shape :code:`(..., size)`. See :func:`transform_irrft`
            for details.
    """
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, size=y.shape[-1])
    return unpack_rfft(np.fft.rfft(y - loc) / rfft_scale, y.shape[-1])


def evaluate_log_prob_rfft(y: np.ndarray, loc: np.ndarray, *,
                           cov_rfft: Optional[np.ndarray] = None,
                           cov: Optional[np.ndarray] = None,
                           rfft_scale: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Evaluate the log probability of a one-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape :code:`(..., size)`, where :code:`...` is
            the batch shape and :code:`size` is the number of grid points.
        loc: Mean of the Gaussian process with shape :code:`(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            :code:`(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape :code:`(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape
            :code:`(..., size // 2 + 1)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape :code:`(...)`.
    """
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, y.shape[-1])
    rfft = transform_rfft(y, loc, rfft_scale=rfft_scale)
    return log_prob_stdnorm(rfft).sum(axis=-1) \
        + evaluate_rfft_log_abs_det_jac(y.shape[-1], rfft_scale=rfft_scale)


def evaluate_rfft_log_abs_det_jac(size: int, *, cov_rfft: Optional[np.ndarray] = None,
                                  cov: Optional[np.ndarray] = None,
                                  rfft_scale: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Evaluate the log absolute determinant of the Jacobian associated with :func:`transform_rfft`.

    Args:
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            :code:`(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape :code:`(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape
            :code:`(..., size // 2 + 1)`.

    Returns:
        log_abs_det_jac: Log absolute determinant of the Jacobian.
    """
    imagidx = (size + 1) // 2
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, size)
    assert rfft_scale.shape[-1] == size // 2 + 1
    return - np.log(rfft_scale).sum(axis=-1) \
        - np.log(rfft_scale[1:imagidx]).sum(axis=-1) - log2 * ((size - 1) // 2) \
        + size * math.log(size) / 2
