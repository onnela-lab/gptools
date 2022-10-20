// Fourier gaussian process with normal noise and centered parameterization.

functions {
    #include gptools_util.stan
    #include gptools_fft1.stan
    #include gptools_kernels.stan
}

#include data.stan

parameters {
    vector[n] eta;
}

model {
    vector[n] cov = gp_periodic_exp_quad_cov(zeros_vector(1), X, sigma, length_scale, n);
    vector[n %/% 2 + 1] rfft_scale = gp_evaluate_rfft_scale(cov);
    eta ~ gp_rfft(zeros_vector(n), rfft_scale);
    y ~ normal(eta, noise_scale);
}
