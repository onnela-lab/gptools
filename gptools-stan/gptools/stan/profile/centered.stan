// Graph gaussian process with normal noise and centered parametrization.

functions {
    #include gptools_graph.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta;
}

model {
    eta ~ graph_gp(X, alpha, rho, epsilon, edge_index, degrees);
    y ~ normal(eta, noise_scale);
}
