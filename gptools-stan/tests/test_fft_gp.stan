functions {
    #include gptools.stan
}

data {
    int n;
    vector[n] y;
    vector[n] cov;
}

generated quantities {
   real log_prob = fft_gp_lpdf(y | cov);
}
