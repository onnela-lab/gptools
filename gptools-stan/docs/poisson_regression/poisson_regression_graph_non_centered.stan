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
    vector[n] eta = gp_transform_inv_graph_exp_quad_cov(
        z, zeros_vector(n), X, sigma, length_scale, edge_index, degrees, epsilon);
}

model {
    // White noise prior (implies eta ~ gp_graph_exp_quad_cov(...)) and observation model.
    z ~ normal(0, 1);
    y ~ poisson_log(eta);
}
