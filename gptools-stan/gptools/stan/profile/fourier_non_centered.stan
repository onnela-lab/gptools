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
        vector[n] cov = gp_periodic_exp_quad_cov(zeros_vector(1), X, sigma, length_scale, n);
        vector[n %/% 2 + 1] rfft_scale = gp_evaluate_rfft_scale(cov);
        eta = gp_transform_irfft(eta_, zeros_vector(n), rfft_scale);
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
