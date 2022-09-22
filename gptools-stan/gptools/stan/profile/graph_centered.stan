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
    eta ~ graph_gp(X, alpha, rho, epsilon, edge_index, degrees);
    y ~ normal(eta, noise_scale);
}
