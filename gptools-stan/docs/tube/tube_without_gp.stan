data {
    int num_stations, num_edges, num_zones, num_degrees;
    array [num_stations] int passengers;
    matrix[num_stations, num_zones] one_hot_zones;
    matrix[num_stations, num_degrees] one_hot_degrees;

    int include_zone_effect;
    int include_degree_effect;
}

parameters {
    real mu;
    real<lower=0> kappa;
    vector[num_zones] zone_effect;
    vector[num_degrees] degree_effect;
}

transformed parameters {
    vector[num_stations] log_mean = mu
        + include_zone_effect * one_hot_zones * zone_effect
        + include_degree_effect * one_hot_degrees * degree_effect;

}

model {
    zone_effect ~ cauchy(0, 1);
    degree_effect ~ cauchy(0, 1);
    kappa ~ cauchy(0, 1);
    for (i in 1:num_stations) {
        if (passengers[i] >= 0) {
            log(passengers[i]) ~ normal(log_mean[i], kappa);
        }
    }
}
