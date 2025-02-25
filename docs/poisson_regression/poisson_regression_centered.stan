// Gaussian process with log link for Poisson observations using the default centered
// parameterization.

data {
    // Shared data definitions.
    #include data.stan
}

parameters {
    // Latent log-rate we seek to infer.
    vector[n] eta;
}

model {
    // Gaussian process prior and observation model. The covariance should technically have periodic
    // boundary conditions to match the generative model in the example.
    matrix[n, n] cov = add_diag(gp_exp_quad_cov(X, X, sigma, length_scale), epsilon);
    eta ~ multi_normal(zeros_vector(n), cov);
    y ~ poisson_log(eta);
}
