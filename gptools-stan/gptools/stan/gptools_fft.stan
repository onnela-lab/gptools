/**
* Evaluate the log probability of a one-dimensional Gaussian process in Fourier space.
*
* @param y Random variable whose likelihood to evaluate.
* @param cov First row of the covariance matrix.
*
* @return Log probability of the Gaussian process.
*/
real fft_gp_lpdf(vector y, vector cov) {
    int n = size(y);
    int m = n %/% 2 + 1;
    // The last index of imaginary components to consider. This is necessary to distinguish between
    // the odd case (without Nyqvist frequency) and even (with Nyqvist frequency).
    int idx;
    // Evaluate the scale of Fourier coefficients.
    vector[m] fft_scale = sqrt(n * get_real(fft(cov)[:m]) / 2);
    // The first element has larger scale because it only has a real part but must still have the
    // right variance. The same applies to the last element if the number of elements is even
    // (Nyqvist frequency).
    fft_scale[1] *= sqrt(2);
    if (n % 2 == 0) {
        fft_scale[m] *= sqrt(2);
        idx = m - 1;
    } else {
        idx = m;
    }
    complex_vector[m] fft = fft(y)[:m];
    return normal_lpdf(get_real(fft) | 0, fft_scale)
        + normal_lpdf(get_imag(fft[2:idx]) | 0, fft_scale[2:idx])
        - log(2) * ((n - 1) %/% 2) + n * log(n) / 2;
}

/**
* Evaluate the log probability of a two-dimensional Gaussian process in Fourier space.
*
* @param y Random variable whose likelihood to evaluate.
* @param cov First row of the covariance matrix.
*
* @return Log probability of the Gaussian process.
*/
real fft2_gp_lpdf(matrix y, matrix cov) {
    array [2] int ydims = dims(y);
    int height = ydims[1];
    int width = ydims[2];
    int n = width * height;
    int fftwidth = width %/% 2 + 1;
    int fftheight = height %/% 2 + 1;

    // Evaluate the Fourier coefficients and their scale. We divide the latter by two to account for
    // real and imaginary parts.
    complex_matrix[height, fftwidth] ffty = fft2(y)[:, :fftwidth];
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
