functions {
    #include gptools_util.stan
}

data {
    int n;
    vector[n] x;
}

generated quantities {
    complex_vector[n] y = fft(x);
    complex_vector[n] z = inv_fft(y);
    assert_close(get_real(z), x);
    assert_close(get_imag(z), 0);
}
