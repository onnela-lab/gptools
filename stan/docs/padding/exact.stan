data {
    int num_observations, observe_first;
    array [num_observations] real x;
    vector[num_observations] y;
    real<lower=0> kappa, length_scale, epsilon, sigma;
}

transformed data {
    cov_matrix[num_observations] cov = add_diag(gp_exp_quad_cov(x, sigma, length_scale), epsilon);
    matrix[num_observations, num_observations] chol = cholesky_decompose(cov);
}

parameters {
    vector[num_observations] z;
}

transformed parameters {
    vector[num_observations] f = chol * z;
}

model {
    z ~ normal(0, 1);
    y[:observe_first] ~ normal(f[:observe_first], kappa);
}
