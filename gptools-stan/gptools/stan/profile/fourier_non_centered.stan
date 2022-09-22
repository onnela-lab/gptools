// Fourier gaussian process with normal noise and non-centered parameterization.

functions {
    #include gptools_fft.stan
    #include gptools_kernels.stan
    #include gptools_util.stan
}

#include data.stan

parameters {
    vector[n] eta_;
}

transformed parameters {
    vector[n] eta;
    {
        vector[n] cov = gp_periodic_exp_quad_cov(zeros(1), X, alpha, rho, n);
        eta = fft_gp_transform(eta_, cov);
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
