functions {
    #include gptools_util.stan
    #include gptools_fft1.stan
}

data {
    int n;
    vector[n] y;
    vector[n] loc;
    vector[n] cov;
}

generated quantities {
   real log_prob = gp_rfft_lpdf(y | loc, cov);
}
