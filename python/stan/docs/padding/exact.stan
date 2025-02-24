data {
    int num_observations, observe_first;
    int kernel;
    array [num_observations] real x;
    vector[num_observations] y;
    real<lower=0> kappa, length_scale, epsilon, sigma;
}

transformed data {
    cov_matrix[num_observations] cov;
    if (kernel == 0) {
        cov = gp_exp_quad_cov(x, sigma, length_scale);
    } else if (kernel == 1) {
        cov = gp_matern32_cov(x, sigma, length_scale);
    } else {
        reject("kernel=", kernel);
    }
    cov = add_diag(cov, epsilon);
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
