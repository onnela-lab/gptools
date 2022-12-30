// Fourier gaussian process with normal noise and centered parameterization.

functions {
    #include gptools/util.stan
    #include gptools/fft1.stan
}

#include data.stan

parameters {
    vector[n] eta;
}

model {
    vector[n %/% 2 + 1] cov_rfft = gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, n);
    eta ~ gp_rfft(zeros_vector(n), cov_rfft);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
