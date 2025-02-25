functions {
    // Include utility functions, such as real fast Fourier transforms.
    #include gptools/util.stan
    // Include functions to evaluate GP likelihoods with Fourier methods.
    #include gptools/fft.stan
}

data {
    // The number of sample points.
    int<lower=1> n;
    // Real fast Fourier transform of the covariance kernel.
    vector[n %/% 2 + 1] cov_rfft;
}

parameters {
    // GP value at the `n` sampling points.
    vector[n] z;
}

transformed parameters {
    // Transform from white noise to a GP realization.
    vector[n] f = gp_inv_rfft(z, zeros_vector(n), cov_rfft);
}

model {
    // Implies that `f` is a GP.
    z ~ normal(0, 1);
}
