data {
    int n;
    vector[n] x;
}

generated quantities {
   complex_vector[n] y = fft(x);
}
