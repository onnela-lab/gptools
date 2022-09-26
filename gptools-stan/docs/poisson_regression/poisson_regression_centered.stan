// Gaussian process with log link for Poisson observations using the default centered
// parameterization.

functions {
    // Include so we can evaluate kernels with periodic boundary conditions and use convenience
    // functions such as `zeros`.
    #include gptools_util.stan
    #include gptools_kernels.stan
}

data {
    // Shared data definitions.
    #include data.stan
}

parameters {
    // Latent log-rate we seek to infer.
    vector[n] eta;
}

model {
    // Gaussian process prior and observation model.
    matrix[n, n] cov = add_diag(gp_periodic_exp_quad_cov(X, sigma, length_scale, n), epsilon);
    eta ~ multi_normal(zeros(n), cov);
    y ~ poisson_log(eta);
}
