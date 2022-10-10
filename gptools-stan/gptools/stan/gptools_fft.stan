// IMPORTANT: stan uses the questionable R indexing which is one-based and inclusive on both ends.
// I.e., x[1:3] includes x[1] to x[3]. More generally, x[i:j] comprises j - i + 1 elements. It could
// at least have been exclusive on the right...

/**
Evaluate the scale of Fourier coefficients.

The Fourier coefficients of a zero-mean Gaussian process with even covariance function are
uncorrelated Gaussian random variables with zero mean. This function evaluates their scale.

:param cov: Covariance between the origin and the rest of the domain.
*/
vector gp_evaluate_rfft_scale(vector cov) {
    int n = size(cov);
    vector[n] result = n * get_real(fft(cov)) / 2;
    if (min(result) < 0){
        reject("kernel is not positive definite");
    }
    // The first element has larger scale because it only has a real part but must still have the
    // right variance. The same applies to the last element if the number of elements is even
    // (Nyqvist frequency).
    result[1] *= 2;
    if (n % 2 == 0) {
        result[n %/% 2 + 1] *= 2;
    }
    return sqrt(result);
}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
vector gp_transform_rfft(vector y, vector loc, vector cov, vector rfft_scale) {
    int n = size(y);
    vector[n] z;
    int ncomplex = (n - 1) %/% 2;
    int nrfft = n %/% 2 + 1;
    complex_vector[n] fft = fft(y - loc) ./ rfft_scale;
    z[1:nrfft] = get_real(fft[1:nrfft]);
    z[1 + nrfft:n] = get_imag(fft[2:1 + ncomplex]);
    return z;
}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
vector gp_transform_rfft(vector y, vector loc, vector cov) {
    return gp_transform_rfft(y, loc, cov, gp_evaluate_rfft_scale(cov));
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :func:`gp_transform_rfft`.
*/
real gp_fft_log_abs_det_jacobian(vector cov, vector rfft_scale) {
    int n = size(rfft_scale);
    return - sum(log(rfft_scale[1:n %/% 2 + 1])) -sum(log(rfft_scale[2:(n + 1) %/% 2]))
        - log(2) * ((n - 1) %/% 2) + n * log(n) / 2;
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :func:`gp_transform_rfft`.
*/
real gp_fft_log_abs_det_jacobian(vector cov) {
    return gp_fft_log_abs_det_jacobian(cov, gp_evaluate_rfft_scale(cov));
}


/**
Evaluate the log probability of a one-dimensional Gaussian process with zero mean in Fourier
space.

:param y: Random variable whose likelihood to evaluate.
:param loc: Mean of the Gaussian process.
:param cov: Covariance between the origin and the rest of the domain (see
    :cpp:func:`gp_evaluate_rfft_scale` for details).

:returns: Log probability of the Gaussian process.
*/
real gp_fft_lpdf(vector y, vector loc, vector cov) {
    int n = size(y);
    vector[n] rfft_scale = gp_evaluate_rfft_scale(cov);
    vector[n] z = gp_transform_rfft(y, loc, cov, rfft_scale);
    return std_normal_lpdf(z) + gp_fft_log_abs_det_jacobian(cov, rfft_scale);
}


/**
Transform white noise in the Fourier domain to a Gaussian process realization, i.e., a
*non-centered* parameterization in the Fourier domain.

The :math:`n` real white noise variables must be assembled into a length-:math:`n` complex vector
with structure expected by the fast Fourier transform. The input vector :math:`z` comprises

- the real zero frequency term,
- :math:`\text{floor}\left(\frac{n - 1}{2}\right)` real parts of positive frequency terms,
- the real Nyqvist frequency term if :math:`n` is even,
- and :math:`\text{floor}\left(\frac{n - 1}{2}\right)` imaginary parts of positive frequency terms.

:param z: Fourier-domain white noise comprising :math:`n` elements.
:param loc: Mean of the Gaussian process.
:param cov: Covariance between the origin and the rest of the domain (see
    :cpp:func:`gp_evaluate_rfft_scale` for details).

:returns: Realization of the Gaussian process with :math:`n` elements.
*/
vector gp_transform_irfft(vector z, vector loc, vector cov) {
    int n = size(z);  // Number of observations.
    int ncomplex = (n - 1) %/% 2;  // Number of complex Fourier coefficients.
    int nrfft = n %/% 2 + 1;  // Number of elements in the real FFT.
    int neg_offset = (n + 1) %/% 2;  // Offset at which the negative frequencies start.
    complex_vector[n] fft;

    // Zero frequency, real part of positive frequency coefficients, and Nyqvist frequency.
    fft[1:nrfft] = z[1:nrfft];
    // Imaginary part of positive frequency coefficients.
    fft[2:ncomplex + 1] += 1.0i * z[nrfft + 1:n];
    // Negative frequency coefficients.
    fft[nrfft + 1:n] = reverse(to_complex(z[2:ncomplex + 1], -z[nrfft + 1:n]));
    return get_real(inv_fft(gp_evaluate_rfft_scale(cov) .* fft)) + loc;
}


/**
Evaluate the log probability of a two-dimensional Gaussian process with zero mean in Fourier space.

:param y: Random variable whose likelihood to evaluate.
:param loc: Mean of the Gaussian process.
:param cov: First row of the covariance matrix.

:returns: Log probability of the Gaussian process.
*/
real gp_fft2_lpdf(matrix y, matrix loc, matrix cov) {
    array [2] int ydims = dims(y);
    int height = ydims[1];
    int width = ydims[2];
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;

    // Evaluate the Fourier coefficients and their scale. We divide the latter by two to account for
    // real and imaginary parts.
    complex_matrix[height, fftwidth] ffty = fft2(y - loc)[:, :fftwidth];
    matrix[height, fftwidth] fftreal = get_real(ffty);
    matrix[height, fftwidth] fftimag = get_imag(ffty);
    matrix[height, fftwidth] fftscale = sqrt(n * get_real(fft2(cov)[:, :fftwidth]) / 2);

    // Adjust the scale for the zero-frequency (and Nyqvist) terms in the first column.
    fftscale[1, 1] *= sqrt2();
    if (height % 2 == 0) {
        fftscale[fftheight, 1] *= sqrt2();
    }
    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;
    real log_prob = normal_lpdf(fftreal[:fftheight, 1] | 0, fftscale[:fftheight, 1])
        + normal_lpdf(fftimag[2:idx, 1] | 0, fftscale[2:idx, 1]);

    // Evaluate the "bulk" likelihood that needs no adjustment.
    log_prob += normal_lpdf(to_vector(fftreal[:, 2:fftwidth - 1]) | 0, to_vector(fftscale[:, 2:fftwidth - 1]))
        + normal_lpdf(to_vector(fftimag[:, 2:fftwidth - 1]) | 0, to_vector(fftscale[:, 2:fftwidth - 1]));

    if (width % 2) {
        // If the width is odd, the last column comprises all-independent terms.
        log_prob += normal_lpdf(fftreal[:, fftwidth] | 0, fftscale[:, fftwidth])
            + normal_lpdf(fftimag[:, fftwidth] | 0, fftscale[:, fftwidth]);
    } else {
        // If the width is even, the last column has the same structure as the first column.
        fftscale[1, fftwidth] *= sqrt2();
        if (height % 2 == 0) {
            fftscale[fftheight, fftwidth] *= sqrt2();
        }
        log_prob += normal_lpdf(fftreal[:fftheight, fftwidth] | 0, fftscale[:fftheight, fftwidth])
            + normal_lpdf(fftimag[2:idx, fftwidth] | 0, fftscale[2:idx, fftwidth]);
    }

    // Correction terms from the transform that only depend on the shape.
    int nterms = (n - 1) %/% 2;
    if (height % 2 == 0 && width % 2 == 0) {
        nterms -=1;
    }
    log_prob += - log2() * nterms + n * log(n) / 2;
    return log_prob;
}
