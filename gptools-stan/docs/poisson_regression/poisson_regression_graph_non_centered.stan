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
    vector[n] z;
}

transformed parameters {
    // Transform white noise to a sample from the graph Gaussian process.
    vector[n] eta = graph_gp_transform(z, X, sigma, length_scale, epsilon, edge_index, degrees);
}

model {
    // White noise prior (implies eta ~ graph_gp(...)) and observation model.
    z ~ normal(0, 1);
    y ~ poisson_log(eta);
}
