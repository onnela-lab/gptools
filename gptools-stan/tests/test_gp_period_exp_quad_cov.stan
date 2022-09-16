functions {
    #include gptools_kernels.stan
}

data {
    int n, m, p;
    array [n] vector[p] x1;
    array [m] vector[p] x2;
    real sigma, epsilon;
    vector[p] length_scale, period;
}

generated quantities {
    matrix[n, m] cov = add_diag(gp_periodic_exp_quad_cov(x1, x2, sigma, length_scale, period),
                                epsilon);
}
