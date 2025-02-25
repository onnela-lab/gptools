functions {
    #include gptools/util.stan
    #include gptools/graph.stan
}

data {
    int num_stations, num_edges, num_zones, num_degrees;
    array [num_stations] vector[2] station_locations;
    array [num_stations] int passengers;
    array [2, num_edges] int edge_index;
    matrix[num_stations, num_zones] one_hot_zones;
    matrix[num_stations, num_degrees] one_hot_degrees;
    int include_zone_effect, include_degree_effect;
}

parameters {
    vector[num_stations] z;
    real loc;
    real<lower=0> sigma, kappa;
    real<lower=log(0.32), upper=log(31)> log_length_scale;
    vector[num_zones] zone_effect;
    vector[num_degrees] degree_effect;
}

transformed parameters {
    real length_scale = exp(log_length_scale);
    vector[num_stations] f = gp_inv_graph_exp_quad_cov(
        z, zeros_vector(num_stations), station_locations, sigma, length_scale, edge_index);
    vector[num_stations] log_mean = loc + f
        + include_zone_effect * one_hot_zones * zone_effect
        + include_degree_effect * one_hot_degrees * degree_effect;
}

model {
    z ~ std_normal();
    sigma ~ student_t(2, 0, 1);
    zone_effect ~ student_t(2, 0, 1);
    degree_effect ~ student_t(2, 0, 1);
    kappa ~ student_t(2, 0, 1);
    for (i in 1:num_stations) {
        if (passengers[i] >= 0) {
            log(passengers[i]) ~ normal(log_mean[i], kappa);
        }
    }
}
