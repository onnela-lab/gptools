// Graph gaussian process with normal noise and centered parameterization.

functions {
    #include gptools_graph.stan
}

#include data.stan

transformed data {
    array [n] int degrees = in_degrees(n, edge_index);
}

parameters {
    vector[n] eta;
}

model {
    eta ~ gp_graph_exp_quad_cov(X, sigma, length_scale, edge_index, degrees, epsilon);
    y[observed_idx] ~ normal(eta[observed_idx], noise_scale);
}
