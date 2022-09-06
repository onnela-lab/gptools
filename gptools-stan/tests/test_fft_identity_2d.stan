data {
    int n, m;
    matrix[n, m] x;
}

generated quantities {
   complex_matrix[n, m] y = fft2(x);
}
