functions {
    #include gptools_fft.stan
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
