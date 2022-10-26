// IMPORTANT: stan uses the questionable R indexing which is one-based and inclusive on both ends.
// I.e., x[1:3] includes x[1] to x[3]. More generally, x[i:j] comprises j - i + 1 elements. It could
// at least have been exclusive on the right...

vector gp_evaluate_rfft_scale(vector rfft_, int n) {
    int nrfft = n %/% 2 + 1;
    vector[nrfft] result = n * rfft_ / 2;
    // Check positive-definiteness.
    real minval = min(result);
    if (minval < 0) {
        reject("covariance matrix is not positive-definite (minimum eigenvalue is ", minval, ")");
    }
    // The first element has larger scale because it only has a real part but must still have the
    // right variance. The same applies to the last element if the number of elements is even
    // (Nyqvist frequency).
    result[1] *= 2;
    if (n % 2 == 0) {
        result[nrfft] *= 2;
    }
    return sqrt(result);
}

/**
Evaluate the scale of Fourier coefficients.

The Fourier coefficients of a zero-mean Gaussian process with even covariance function are
uncorrelated Gaussian random variables with zero mean. This function evaluates their scale.

:param cov: Covariance between the origin and the rest of the domain.
*/
vector gp_evaluate_rfft_scale(vector cov) {
    return gp_evaluate_rfft_scale(get_real(rfft(cov)), size(cov));
}


/*
Unpack the complex Fourier coefficients of a real Fourier transform with `n` elements to a vector of
`n` elements.
*/
vector gp_unpack_rfft(complex_vector x, int n) {
    vector[n] z;
    int ncomplex = (n - 1) %/% 2;
    int nrfft = n %/% 2 + 1;
    z[1:nrfft] = get_real(x);
    z[1 + nrfft:n] = get_imag(x[2:1 + ncomplex]);
    return z;
}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
vector gp_transform_rfft(vector y, vector loc, vector rfft_scale) {
    return gp_unpack_rfft(rfft(y - loc) ./ rfft_scale, size(y));
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :stan:func:`gp_transform_rfft`.
*/
real gp_rfft_log_abs_det_jacobian(vector rfft_scale, int n) {
    return - sum(log(rfft_scale[1:n %/% 2 + 1])) -sum(log(rfft_scale[2:(n + 1) %/% 2]))
        - log(2) * ((n - 1) %/% 2) + n * log(n) / 2;
}


/**
Evaluate the log probability of a one-dimensional Gaussian process with zero mean in Fourier
space.

:param y: Random variable whose likelihood to evaluate.
:param loc: Mean of the Gaussian process.
:param cov: Covariance between the origin and the rest of the domain (see
    :stan:func:`gp_evaluate_rfft_scale(vector)` for details).

:returns: Log probability of the Gaussian process.
*/
real gp_rfft_lpdf(vector y, vector loc, vector rfft_scale) {
    int n = size(y);
    int nrfft = n %/% 2 + 1;
    vector[n] z = gp_transform_rfft(y, loc, rfft_scale);
    return std_normal_lpdf(z) + gp_rfft_log_abs_det_jacobian(rfft_scale, n);
}


/*
Transform a real vector with `n` elements to a vector of complex Fourier coefficients with `n`
elements ready for inverse real fast Fourier transformation.
*/
complex_vector gp_pack_rfft(vector z) {
    int n = size(z);  // Number of observations.
    int ncomplex = (n - 1) %/% 2;  // Number of complex Fourier coefficients.
    int nrfft = n %/% 2 + 1;  // Number of elements in the real FFT.
    int neg_offset = (n + 1) %/% 2;  // Offset at which the negative frequencies start.
    // Zero frequency, real part of positive frequency coefficients, and Nyqvist frequency.
    complex_vector[nrfft] fft = z[1:nrfft];
    // Imaginary part of positive frequency coefficients.
    fft[2:ncomplex + 1] += 1.0i * z[nrfft + 1:n];
    return fft;
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
    :stan:func:`gp_evaluate_rfft_scale(vector)` for details).

:returns: Realization of the Gaussian process with :math:`n` elements.
*/
vector gp_transform_irfft(vector z, vector loc, vector rfft_scale) {
    return get_real(inv_rfft(rfft_scale .* gp_pack_rfft(z), size(z))) + loc;
}
