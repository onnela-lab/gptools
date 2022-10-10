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


/*
Unpack the complex Fourier coefficients of a real Fourier transform with `n` elements to a vector of
`n` elements.
*/
vector gp_unpack_rfft(complex_vector rfft) {
    int n = size(rfft);
    vector[n] z;
    int ncomplex = (n - 1) %/% 2;
    int nrfft = n %/% 2 + 1;
    z[1:nrfft] = get_real(rfft[1:nrfft]);
    z[1 + nrfft:n] = get_imag(rfft[2:1 + ncomplex]);
    return z;
}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
vector gp_transform_rfft(vector y, vector loc, vector cov, vector rfft_scale) {
    return gp_unpack_rfft(fft(y - loc) ./ rfft_scale);

}


/**
Transform a Gaussian process realization to white noise in the Fourier domain.
*/
vector gp_transform_rfft(vector y, vector loc, vector cov) {
    return gp_transform_rfft(y, loc, cov, gp_evaluate_rfft_scale(cov));
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :cpp:func:`gp_transform_rfft`.
*/
real gp_fft_log_abs_det_jacobian(vector cov, vector rfft_scale) {
    int n = size(rfft_scale);
    return - sum(log(rfft_scale[1:n %/% 2 + 1])) -sum(log(rfft_scale[2:(n + 1) %/% 2]))
        - log(2) * ((n - 1) %/% 2) + n * log(n) / 2;
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :cpp:func:`gp_transform_rfft`.
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


/*
Transform a real vector with `n` elements to a vector of complex Fourier coefficients with `n`
elements ready for inverse real fast Fourier transformation.
*/
complex_vector gp_pack_rfft(vector z) {
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
    :cpp:func:`gp_evaluate_rfft_scale` for details).

:returns: Realization of the Gaussian process with :math:`n` elements.
*/
vector gp_transform_irfft(vector z, vector loc, vector cov) {
    return get_real(inv_fft(gp_evaluate_rfft_scale(cov) .* gp_pack_rfft(z))) + loc;
}


/**
Evaluate the scale of Fourier coefficients.
*/
matrix gp_evaluate_rfft2_scale(matrix cov) {
    array [2] int covdims = dims(cov);
    int height = covdims[1];
    int width = covdims[2];
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    matrix[height, fftwidth] fftscale = n * get_real(fft2(cov)[:, :fftwidth]) / 2;

    // Adjust the scale for the zero-frequency (and Nyqvist) terms in the first column.
    fftscale[1, 1] *= 2;
    if (height % 2 == 0) {
        fftscale[fftheight, 1] *= 2;
    }
    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;

    if (width % 2 == 0) {
        // If the width is even, the last column has the same structure as the first column.
        fftscale[1, fftwidth] *= 2;
        if (height % 2 == 0) {
            fftscale[fftheight, fftwidth] *= 2;
        }
    }
    return sqrt(fftscale);
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :cpp:func:`gp_transform_rfft`.
*/
real gp_fft2_log_abs_det_jacobian(matrix cov, matrix fftscale) {
    array [2] int ydims = dims(cov);
    int height = ydims[1];
    int width = ydims[2];
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;
    real ladj = 0;

    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;
    ladj += - sum(log(fftscale[:fftheight, 1])) - sum(log(fftscale[2:idx, 1]));

    // Evaluate the "bulk" likelihood that needs no adjustment.
    ladj += - 2 * sum(log(to_vector(fftscale[:, 2:fftwidth - 1])));

    if (width % 2) {
        // If the width is odd, the last column comprises all-independent terms.
        ladj += -sum(log(fftscale[:, fftwidth])) - sum(log(fftscale[:, fftwidth]));
    } else {
        ladj += -sum(log(fftscale[:fftheight, fftwidth])) - sum(log(fftscale[2:idx, fftwidth]));
    }
    // Correction terms from the transform that only depend on the shape.
    int nterms = (n - 1) %/% 2;
    if (height % 2 == 0 && width % 2 == 0) {
        nterms -=1;
    }
    ladj += - log2() * nterms + n * log(n) / 2;
    return ladj;
}


/**
Evaluate the log absolute determinant of the Jacobian associated with :cpp:func:`gp_transform_rfft`.
*/
real gp_fft2_log_abs_det_jacobian(matrix cov) {
    return gp_fft2_log_abs_det_jacobian(cov, gp_evaluate_rfft2_scale(cov));
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
    matrix[height, fftwidth] fftscale = gp_evaluate_rfft2_scale(cov);
    complex_matrix[height, fftwidth] ffty = fft2(y - loc)[:, :fftwidth] ./ fftscale;
    matrix[height, fftwidth] fftreal = get_real(ffty);
    matrix[height, fftwidth] fftimag = get_imag(ffty);


    // For the real part, we always use the full height of the non-redundant part. For the imaginary
    // part, we discard the last element if the number of rows is even because it's the real Nyqvist
    // frequency.
    int idx = (height % 2) ? fftheight : fftheight - 1;
    real log_prob = normal_lpdf(fftreal[:fftheight, 1] | 0, 1)
        + normal_lpdf(fftimag[2:idx, 1] | 0, 1);

    // Evaluate the "bulk" likelihood that needs no adjustment.
    log_prob += std_normal_lpdf(to_vector(fftreal[:, 2:fftwidth - 1]))
        + std_normal_lpdf(to_vector(fftimag[:, 2:fftwidth - 1]));

    if (width % 2) {
        // If the width is odd, the last column comprises all-independent terms.
        log_prob += std_normal_lpdf(fftreal[:, fftwidth])
            + std_normal_lpdf(fftimag[:, fftwidth]);
    } else {
        log_prob += std_normal_lpdf(fftreal[:fftheight, fftwidth])
            + std_normal_lpdf(fftimag[2:idx, fftwidth]);
    }

    return log_prob + gp_fft2_log_abs_det_jacobian(cov, fftscale);
}
