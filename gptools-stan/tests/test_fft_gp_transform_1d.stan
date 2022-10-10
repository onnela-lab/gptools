functions {
    #include gptools_util.stan
    #include gptools_fft1.stan
}

data {
    int n;
    vector[n] z;
    vector[n] loc;
    vector[n] cov;
}

generated quantities {
    vector[n] y = gp_transform_irfft(z, loc, cov);
}
