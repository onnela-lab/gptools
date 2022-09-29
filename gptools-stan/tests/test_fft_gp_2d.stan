functions {
    #include gptools_fft.stan
}

data {
    int n, m;
    matrix[n, m] y;
    matrix[n, m] cov;
}

generated quantities {
    real log_prob = gp_fft2_lpdf(y | cov);
}
