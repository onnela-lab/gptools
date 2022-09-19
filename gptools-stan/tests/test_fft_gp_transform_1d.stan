functions {
    #include gptools_fft.stan
}

data {
    int n;
    vector[n] z;
    vector[n] cov;
}

generated quantities {
    vector[n] y = fft_gp_transform(z, cov);
}
