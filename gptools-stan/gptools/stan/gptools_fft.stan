/**
* Evaluate the log probability of a Gaussian process in Fourier space.
*/
real fft_gp_lpdf(vector y, vector cov) {
    int n = size(y);
    int m = n %/% 2 + 1;
    // The last index of imaginary components to consider. This is necessary to distinguish between
    // the odd case (without Nyqvist frequency) and even (with Nyqvist frequency).
    int idx;
    // Evaluate the scale of Fourier coefficients.
    vector[m] fft_scale = sqrt(n * abs(fft(cov)[:m]) / 2);
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
