// Fourier gaussian process with normal noise and non-centered parameterization.

functions {
    #include gptools_util.stan
    #include gptools_fft1.stan
    #include gptools_kernels.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    vector[n] eta;
    {
        vector[n %/% 2 + 1] rfft_cov = gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, n, 10);
        vector[n %/% 2 + 1] rfft_scale = gp_evaluate_rfft_scale(rfft_cov, n);
        eta = gp_transform_irfft(eta_, zeros_vector(n), rfft_scale);
    }
}

model {
    eta_ ~ normal(0, 1);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
