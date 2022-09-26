// Graph gaussian process with log link for Poisson observations.

functions {
    #include gptools_graph.stan
}

data {
    #include data.stan
    // Information about the graph.
    int num_edges;
    array [2, num_edges] int edge_index;
}

transformed data {
    array [n] int degrees = in_degrees(n, edge_index);
}

parameters {
    vector[n] eta;
}

model {
    // Graph Gaussian process prior and observation model.
    eta ~ graph_gp(X, sigma, length_scale, epsilon, edge_index, degrees);
    y ~ poisson_log(eta);
}
