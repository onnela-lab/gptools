// Shared data definition for Poisson regression models.

// Kernel parameters.
real<lower=0> sigma, length_scale, epsilon;

// Information about observation locations and counts.
int n;
array [n] int y;
array [n] vector[1] X;
