functions {
    #include gptools_graph.stan
}

data {
    int num_nodes;
    array [num_nodes] vector[2] X;
    array [num_nodes] int y;
    real<lower=0> epsilon;
    int num_edges;
    array [2, num_edges] int edge_index;

    int num_zones;
    matrix[num_nodes, num_zones] one_hot_zones;
    int include_zone_effect;

    int num_degrees;
    matrix[num_nodes, num_degrees] one_hot_degrees;
    int include_degree_effect;
}

transformed data {
    array [num_nodes] int degrees = in_degrees(num_nodes, edge_index);
}

parameters {
    vector[num_nodes] eta_;
    real<lower=0> sigma;
    real mu;
    real<lower=0> phi;
    vector[num_zones] zone_effect;
    vector[num_degrees] degree_effect;
    real<lower=0> length_scale;
}

transformed parameters {
    vector[num_nodes] eta = graph_gp_transform(eta_, X, sigma, length_scale, epsilon, edge_index, degrees);
    vector[num_nodes] rate = exp(
        mu + eta
        + include_zone_effect * one_hot_zones * zone_effect
        + include_degree_effect * one_hot_degrees * degree_effect
    );

}

model {
    eta_ ~ normal(0, 1);
    for (i in 1:num_nodes) {
        if (y[i] >= 0) {
            y[i] ~ neg_binomial_2(rate[i], 1 / phi);
        }
    }
    phi ~ cauchy(0, 1);
    zone_effect ~ normal(0, 1);
    degree_effect ~ normal(0, 1);
    length_scale ~ inv_gamma(2.0, 2.5);
}
