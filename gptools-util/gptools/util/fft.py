import math
from . import ArrayOrTensor, ArrayOrTensorDispatch


dispatch = ArrayOrTensorDispatch()
sqrt2 = 1.4142135623730951
log2 = 0.6931471805599453
log2pi = 1.8378770664093453


def log_prob_norm(y: ArrayOrTensor, loc: ArrayOrTensor, scale: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the log probability of a normal random variable.
    """
    residual = (y - loc) / scale
    return - dispatch.log(scale) - (log2pi + residual * residual) / 2


def evaluate_rfft_scale(cov: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the scale of Fourier coefficients.

    Args:
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n)`.

    Returns:
        scale: Scale of Fourier coefficients with shape `(..., n // 2 + 1)`.
    """
    *_, size = cov.shape
    scale: ArrayOrTensor = dispatch.sqrt(size * dispatch[cov].fft.rfft(cov).real / 2)
    # Rescale for the real-only zero frequency term.
    scale[0] *= sqrt2
    if size % 2 == 0:
        # Rescale for the real-only Nyqvist frequency term.
        scale[..., -1] *= sqrt2
    return scale


def evaluate_log_prob_rfft(y: ArrayOrTensor, cov: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the log probability of a one-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape `(..., n)`, where `...` is the batch shape
            and `n` is the number of grid points.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape `(...)`.
    """
    *_, size = y.shape
    y = dispatch[y].fft.rfft(y)
    scale = evaluate_rfft_scale(cov)
    imagidx = (size + 1) // 2

    return log_prob_norm(y.real, 0, scale).sum(axis=-1) \
        + log_prob_norm(y.imag[..., 1:imagidx], 0, scale[..., 1:imagidx]).sum(axis=-1) \
        - log2 * ((size - 1) // 2) + size * math.log(size) / 2


def transform_rfft(z: ArrayOrTensor, cov: ArrayOrTensor) -> ArrayOrTensor:
    """
    Transform white noise to a Gaussian process realization.

    Args:
        z: Fourier-domain white noise with shape `(..., size)`. The elements of the white noise
            comprise the `size // 2 + 1` real parts of the zero frequency term, complex terms, and
            Nyqvist frequency term (for even `size`). The subsequent elements are the imaginary
            parts of complex coefficients.
        cov: First row of the covariance matrix with shape `(..., size)`.

    Returns:
        y: Realization of the Gaussian process.
    """
    *_, size = z.shape
    fftsize = size // 2 + 1
    ncomplex = (size - 1) // 2
    # Zero frequency term, real parts of complex coefficients and possible Nyqvist frequency.
    fft = z[..., :fftsize] * (1 + 0j)
    # Imaginary parts of complex coefficients.
    fft[..., 1:ncomplex + 1] += 1j * z[..., fftsize:]
    fft = evaluate_rfft_scale(cov) * fft
    return dispatch[fft].fft.irfft(fft, size)


def transform_irfft(y: ArrayOrTensor, cov: ArrayOrTensor) -> ArrayOrTensor:
    """
    Transform a Gaussian process realization to white noise.

    Args:
        y: Realization of the Gaussian process.
        cov: First row of the covariance matrix.

    Returns:
        z: Fourier-domain white noise.
    """
    # Take the Fourier transform and rescale.
    fft: ArrayOrTensor = dispatch[y].fft.rfft(y) / evaluate_rfft_scale(cov)
    ncomplex = (y.shape[-1] - 1) // 2
    parts = [fft.real, fft.imag[1: ncomplex + 1]]
    if dispatch.is_tensor(y):
        return dispatch[y].concat(parts, axis=-1)
    else:
        return dispatch[y].concatenate(parts, axis=-1)


def evaluate_log_prob_rfft2(y: ArrayOrTensor, cov: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the log probability of a two-dimensional Gaussian process realization in Fourier space.

    Args:
        y: Realization of a Gaussian process with shape `(..., n, m)`, where `...` is the batch
            shape, `n` is the number of rows, and `m` is the number of columns.
        cov: Covariance between the first grid point and the remainder of the grid with shape
            `(..., n, m)`.

    Returns:
        log_prob: Log probability of the Gaussian process realization with shape `(...)`.
    """
    *_, height, width = y.shape
    size = height * width

    # Evaluate the scales Fourier coefficients and their scales. The division by two accounts for
    # half of the variance going to real and imaginary terms each. We subsequently make adjustments
    # based on which elements are purely real.
    ffts = dispatch[y].fft.rfft2(y)
    fftscale = dispatch.sqrt(size * dispatch[cov].fft.rfft2(cov).real / 2)

    # We also construct a binary mask for which elements should be
    # included in the likelihood evaluation.
    fftshape = (height, width // 2 + 1)
    imask = dispatch[y].ones(fftshape)
    rmask = dispatch[y].ones(fftshape)

    # Recall how the two-dimensional RFFT is computed. We first take an RFFT of rows of the matrix.
    # This leaves us with a real first column (zero frequency term) and a real last column if the
    # number of columns is even (Nyqvist frequency term). Second, we take a *full* FFT of the
    # columns. The first column will have a real coefficient in the first row (zero frequency in the
    # "row-dimension"). All elements in rows beyond n // 2 + 1 are irrelevant because the column was
    # real. The same applies to the last column if there is a Nyqvist frequency term. Finally, we
    # will also have a real-only Nyqvist frequency term in the first and last column if the number
    # of rows is even.

    # The first is the zero-frequency term in both dimensions which must always be real. We mask out
    # the last elements of the first column because they are redundant (because the first column is
    # real after the column FFT).
    fftscale[..., 0, 0] *= sqrt2
    imask[0, 0] = 0
    imask[height // 2 + 1:, 0] = 0
    rmask[height // 2 + 1:, 0] = 0

    # If the width is even, we get a real last column after the first transform due to the Nyqvist
    # frequency.
    if width % 2 == 0:
        fftscale[..., 0, -1] *= sqrt2
        imask[0, -1] = 0
        imask[height // 2 + 1:, -1] = 0
        rmask[height // 2 + 1:, -1] = 0

    # If the height is even, we get an extra Nyqvist frequency term in the first column.
    if height % 2 == 0:
        fftscale[..., height // 2, 0] *= sqrt2
        imask[height // 2, 0] = 0

    # If the height and width are even, the Nyqvist frequencies in the last column must be real.
    if width % 2 == 0 and height % 2 == 0:
        fftscale[..., height // 2, -1] *= sqrt2
        imask[height // 2, -1] = 0

    nterms = (size - 1) // 2
    if height % 2 == 0 and width % 2 == 0:
        nterms -= 1
    return (log_prob_norm(ffts.real, 0, fftscale) * rmask).sum(axis=(-1, -2)) + \
        (log_prob_norm(ffts.imag, 0, fftscale) * imask).sum(axis=(-1, -2)) \
        - log2 * nterms + size * math.log(size) / 2
