functions {
    #include gptools_util.stan
    #include gptools_fft.stan
}

data {
    int n;
    vector[n] y;
    vector[n] loc;
    vector[n] cov;
}

generated quantities {
   real log_prob = gp_fft_lpdf(y | loc, cov);
}
