functions {
    #include gptools_util.stan
}

data {
    int n, m;
    matrix[n, m] x;
}

transformed data {
    int mrfft = m %/% 2 + 1;
}

generated quantities {
    // Full Fourier transform.
    complex_matrix[n, m] y = fft2(x);
    complex_matrix[n, m] z = inv_fft2(y);
    assert_close(get_real(z), x);
    assert_close(get_imag(z), 0);

    // Real Fourier transform.
    complex_matrix[n, mrfft] ry = rfft2(x);
    matrix[n, m] rz = inv_rfft2(ry, m);
    assert_close(rz, x);
}
