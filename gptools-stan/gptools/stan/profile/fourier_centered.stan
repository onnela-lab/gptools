// Fourier gaussian process with normal noise and centered parameterization.

functions {
    #include gptools_kernels.stan
    #include gptools_fft.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta;
}

model {
    vector[num_nodes] cov = gp_periodic_exp_quad_cov(X, rep_array(rep_vector(0, 1), 1), alpha, rep_vector(rho, 1), rep_vector(num_nodes, 1))[:, 1];
    eta ~ fft_gp(cov);
    y ~ normal(eta, noise_scale);
}
