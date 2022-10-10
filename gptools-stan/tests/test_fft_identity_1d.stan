functions {
    #include gptools_util.stan
}

data {
    int n;
    vector[n] x;
}

transformed data {
    int nrfft = n %/% 2 + 1;
}

generated quantities {
    // Full Fourier transform.
    complex_vector[n] y = fft(x);
    complex_vector[n] z = inv_fft(y);
    assert_close(get_real(z), x);
    assert_close(get_imag(z), 0);

    // Real Fourier transform.
    complex_vector[nrfft] ry = rfft(x);
    assert_finite(ry);
    vector[n] rz = inv_rfft(ry, n);
    assert_close(get_real(rz), x);
}
