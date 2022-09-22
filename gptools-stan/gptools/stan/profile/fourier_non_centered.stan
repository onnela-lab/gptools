// Fourier gaussian process with normal noise and non-centered parameterization.

functions {
    #include gptools_kernels.stan
    #include gptools_fft.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta_;
}

transformed parameters {
    vector[num_nodes] eta;
    {
        vector[num_nodes] cov = gp_periodic_exp_quad_cov(X, rep_array(rep_vector(0, 1), 1), alpha, rep_vector(rho, 1), rep_vector(num_nodes, 1))[:, 1];
        eta = fft_gp_transform(eta_, cov);
    }
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
