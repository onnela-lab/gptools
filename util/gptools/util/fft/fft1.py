import math
from typing import Optional
from .. import ArrayOrTensor, ArrayOrTensorDispatch, mutually_exclusive_kwargs, \
    OptionalArrayOrTensor
from .util import log2, log_prob_stdnorm, sqrt2


dispatch = ArrayOrTensorDispatch()


def _get_rfft_scale(cov_rfft: OptionalArrayOrTensor, cov: OptionalArrayOrTensor,
                    rfft_scale: OptionalArrayOrTensor, size: Optional[int]) -> ArrayOrTensor:
    num_given = sum([cov_rfft is not None, cov is not None, rfft_scale is not None])
    if num_given != 1:  # pragma: no cover
        raise ValueError("exactly one of `cov_rfft`, `cov`, or `rfft_scale` must be given")
    if rfft_scale is not None:
        return rfft_scale
    return evaluate_rfft_scale(cov_rfft=cov_rfft, cov=cov, size=size)


@mutually_exclusive_kwargs("cov", "cov_rfft")
def evaluate_rfft_scale(*, cov: OptionalArrayOrTensor = None,
                        cov_rfft: OptionalArrayOrTensor = None, size: Optional[int] = None) \
        -> ArrayOrTensor:
    """
    Evaluate the scale of Fourier coefficients.

    Args:
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        size: Size of the real signal. Necessary because the size cannot be inferred from `rfft`.

    Returns:
        scale: Scale of Fourier coefficients with shape `(..., size // 2 + 1)`.
    """
    if cov_rfft is None:
        *_, size = cov.shape
        cov_rfft = dispatch[cov].fft.rfft(cov).real
    scale: ArrayOrTensor = dispatch.sqrt(size * cov_rfft / 2)
    # Rescale for the real-only zero frequency term.
    scale[0] *= sqrt2
    if size % 2 == 0:
        # Rescale for the real-only Nyqvist frequency term.
        scale[..., -1] *= sqrt2
    return scale


def expand_rfft(rfft: ArrayOrTensor, n: int) -> ArrayOrTensor:
    """
    Convert truncated real Fourier coefficients to full Fourier coefficients.

    Args:
        rfft: Truncated real Fourier coefficients with shape `(n // 2 + 1,)`.
        n: Number of samples.

    Returns:
        fft: Full Fourier coefficients with shape `(n,)`.
    """
    nrfft = n // 2 + 1
    ncomplex = (n - 1) // 2
    fft = dispatch[rfft].empty(n, dtype=rfft.dtype)
    fft[:nrfft] = rfft
    fft[nrfft:] = dispatch.flip(rfft[1:1 + ncomplex]).conj()
    return fft


def unpack_rfft(z: ArrayOrTensor, size: int) -> ArrayOrTensor:
    """
    Unpack the Fourier coefficients of a real Fourier transform with `size // 2 + 1` elements to a
    vector of `size` elements.

    Args:
        z: Real Fourier transform coefficients.
        size: Size of the real signal. Necessary because the size cannot be inferred from `rfft`.

    Returns:
        z: Unpacked vector of `size` elements comprising the `size // 2 + 1` real parts of the zero
            frequency term, complex terms, and Nyqvist frequency term (for even `size`). The
            subsequent `(size - 1) // 2` elements are the imaginary parts of complex coefficients.
    """
    ncomplex = (size - 1) // 2
    parts = [z.real, z.imag[..., 1: ncomplex + 1]]
    return dispatch.concatenate(parts, axis=-1)


def pack_rfft(z: ArrayOrTensor, full_fft: bool = False) -> ArrayOrTensor:
    """
    Transform a real vector with `size` elements to a vector of complex Fourier coefficients with
    `size // 2 + 1` elements ready for inverse real fast Fourier transformation.

    Args:
        z: Unpacked vector of `size` elements. See :func:`unpack_rfft` for details.
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
    # Add the redundant complex coefficients (use `flip` because torch does not support negative
    # strides).
    return dispatch.concatenate([rfft, dispatch.flip(rfft[..., 1:ncomplex + 1].conj(), (-1,))], -1)


def transform_irfft(z: ArrayOrTensor, loc: ArrayOrTensor, *, cov_rfft: OptionalArrayOrTensor = None,
                    cov: OptionalArrayOrTensor = None, rfft_scale: OptionalArrayOrTensor = None) \
        -> ArrayOrTensor:
    """
    Transform white noise in the Fourier domain to a Gaussian process realization.

    Args:
        z: Fourier-domain white noise with shape `(..., size)`. See :func:`unpack_rfft` for details.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.

    Returns:
        y: Realization of the Gaussian process with shape `(..., size)`.
    """
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, z.shape[-1])
    rfft = pack_rfft(z) * rfft_scale
    return dispatch[rfft].fft.irfft(rfft, z.shape[-1]) + loc


def transform_rfft(y: ArrayOrTensor, loc: ArrayOrTensor, *, cov_rfft: OptionalArrayOrTensor = None,
                   cov: OptionalArrayOrTensor = None, rfft_scale: OptionalArrayOrTensor = None) \
        -> ArrayOrTensor:
    """
    Transform a Gaussian process realization to white noise in the Fourier domain.

    Args:
        y: Realization of the Gaussian process with shape `(..., size)`.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.

    Returns:
        z: Fourier-domain white noise with shape `(..., size)`. See :func:`transform_irrft` for
            details.
    """
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, size=y.shape[-1])
    return unpack_rfft(dispatch[y].fft.rfft(y - loc) / rfft_scale, y.shape[-1])


def evaluate_log_prob_rfft(y: ArrayOrTensor, loc: ArrayOrTensor, *,
                           cov_rfft: OptionalArrayOrTensor = None,
                           cov: OptionalArrayOrTensor = None,
                           rfft_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log probability of a one-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape `(..., size)`, where `...` is the batch
            shape and `size` is the number of grid points.
        loc: Mean of the Gaussian process with shape `(..., size)`.
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape `(...)`.
    """
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, y.shape[-1])
    rfft = transform_rfft(y, loc, rfft_scale=rfft_scale)
    return log_prob_stdnorm(rfft).sum(axis=-1) \
        + evaluate_rfft_log_abs_det_jacobian(y.shape[-1], rfft_scale=rfft_scale)


def evaluate_rfft_log_abs_det_jacobian(size: int, *, cov_rfft: OptionalArrayOrTensor = None,
                                       cov: OptionalArrayOrTensor = None,
                                       rfft_scale: OptionalArrayOrTensor = None) -> ArrayOrTensor:
    """
    Evaluate the log absolute determinant of the Jacobian associated with :func:`transform_rfft`.

    Args:
        cov_rfft: Precomputed real fast Fourier transform of the kernel with shape
            `(..., size // 2 + 1)`.
        cov: First row of the covariance matrix with shape `(..., size)`.
        rfft_scale: Precomputed real fast Fourier transform scale with shape `(..., size // 2 + 1)`.

    Returns:
        log_abs_det_jacobian: Log absolute determinant of the Jacobian.
    """
    imagidx = (size + 1) // 2
    rfft_scale = _get_rfft_scale(cov_rfft, cov, rfft_scale, size)
    assert rfft_scale.shape[-1] == size // 2 + 1
    return - dispatch.log(rfft_scale).sum(axis=-1) \
        - dispatch.log(rfft_scale[1:imagidx]).sum(axis=-1) - log2 * ((size - 1) // 2) \
        + size * math.log(size) / 2
