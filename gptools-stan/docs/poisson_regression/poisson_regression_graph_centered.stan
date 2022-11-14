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
    eta ~ gp_graph_exp_quad_cov(zeros_vector(n), X, sigma, length_scale, edge_index, degrees,
                                epsilon);
    y ~ poisson_log(eta);
}
