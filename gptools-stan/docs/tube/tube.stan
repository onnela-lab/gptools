functions {
    #include gptools_graph.stan
}

data {
    int num_stations, num_edges, num_zones, num_degrees;
    array [num_stations] vector[2] station_locations;
    array [num_stations] int passengers;
    array [2, num_edges] int edge_index;
    matrix[num_stations, num_zones] one_hot_zones;
    matrix[num_stations, num_degrees] one_hot_degrees;
    real<lower=0> epsilon;

    int include_zone_effect;
    int include_degree_effect;
}

transformed data {
    array [num_stations] int degrees = in_degrees(num_stations, edge_index);
}

parameters {
    vector[num_stations] z;
    real mu;
    real<lower=0> sigma, length_scale, kappa;
    vector[num_zones] zone_effect;
    vector[num_degrees] degree_effect;
}

transformed parameters {
    vector[num_stations] f = gp_transform_inv_graph_exp_quad_cov(
        z, zeros_vector(num_stations), station_locations, sigma, length_scale, edge_index, degrees, epsilon);
    vector[num_stations] log_mean = mu + f
        + include_zone_effect * one_hot_zones * zone_effect
        + include_degree_effect * one_hot_degrees * degree_effect;

}

model {
    z ~ std_normal();
    sigma ~ student_t(2, 0, 1);
    length_scale ~ inv_gamma(2.0, 2.5);
    zone_effect ~ student_t(2, 0, 1);
    degree_effect ~ student_t(2, 0, 1);
    kappa ~ student_t(2, 0, 1);
    for (i in 1:num_stations) {
        if (passengers[i] >= 0) {
            log(passengers[i]) ~ normal(log_mean[i], kappa);
        }
    }
}
