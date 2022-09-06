// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools_graph.stan
}

#include data.stan

parameters {
    vector[num_nodes] eta;
}

model {
    eta ~ graph_gp(X, alpha, rho, epsilon, edge_index, degrees);
    y ~ poisson_log(eta);
}
