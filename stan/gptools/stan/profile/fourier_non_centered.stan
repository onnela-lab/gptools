// Fourier gaussian process with normal noise and non-centered parameterization.

functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    vector[n] eta;
    {
        vector[n %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, n);
        eta = gp_inv_rfft(eta_, zeros_vector(n), cov_rfft);
    }
}

model {
    eta_ ~ normal(0, 1);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
