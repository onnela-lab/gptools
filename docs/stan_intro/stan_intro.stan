data {
    int n, p;
    matrix [n, p] X;
    vector[n] y;
}

parameters {
    vector[p] theta;
    real<lower=0> sigma;
}

model {
    y ~ normal(X * theta, sigma);
    theta ~ normal(0, 1);
    sigma ~ gamma(2, 2);
}
