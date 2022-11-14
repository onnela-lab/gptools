// Graph gaussian process with normal noise and non-centered parameterization.

functions {
    #include gptools_graph.stan
}

#include data.stan

transformed data {
    array [n] int degrees = in_degrees(n, edge_index);
}

parameters {
    vector[n] eta_;
}

transformed parameters {
   vector[n] eta = gp_graph_exp_quad_cov_transform(
    eta_, zeros_vector(n), X, sigma, length_scale, edge_index, degrees, epsilon);
}

model {
    eta_ ~ normal(0, 1);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
