// Graph gaussian process with log link for Poisson observations.

functions {
    #include graph_gaussian_process.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta_;
}

transformed parameters {
    vector[num_nodes] eta = graph_gp_transform(eta_, X, alpha, rho, epsilon, edge_index, degrees);
}

model {
    eta_ ~ normal(0, 1);
    y ~ poisson_log(eta);
}
