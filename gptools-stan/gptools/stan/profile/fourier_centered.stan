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
    vector[n] cov = gp_periodic_exp_quad_cov(zeros(1), X, alpha, rho, n);
    eta ~ gp_rfft(zeros(n), cov);
    y ~ normal(eta, noise_scale);
}
