// Graph gaussian process with log link for Poisson observations.

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
    vector[n] eta = graph_gp_transform(eta_, X, alpha, rho, epsilon, edge_index, degrees);
}

model {
    eta_ ~ normal(0, 1);
    y ~ poisson_log(eta);
}
