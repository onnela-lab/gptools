// Graph gaussian process with normal noise and non-centered parametrization.

functions {
    #include gptools_graph.stan
}

#include data.stan

transformed data {
    array [num_nodes] int degrees = in_degrees(num_nodes, edge_index);
}

parameters {
    vector[num_nodes] eta_;
}

transformed parameters {
   vector[num_nodes] eta = graph_gp_transform(eta_, X, alpha, rho, epsilon, edge_index, degrees);
}

model {
    eta_ ~ normal(0, 1);
    y ~ normal(eta, noise_scale);
}
