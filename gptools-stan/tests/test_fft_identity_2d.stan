data {
    int n, m;
    matrix[n, m] x;
}

generated quantities {
   complex_matrix[n, m] y = fft2(x);
   complex_matrix[n, m] z = inv_fft2(y);
}
