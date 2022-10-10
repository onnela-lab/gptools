// Scalars -----------------------------------------------------------------------------------------

/**
Assert that two integers are equal.

:param actual: Actual value.
:param desired: Desired value.
*/
void assert_equal(int actual, int desired) {
    if (actual != desired) {
        reject(actual, " is not equal to ", desired);
    }
}


/**
Check whether two values are close.

The actual value :math:`x` and desired value :math:`y` may differ by at most :math:`\theta=r y + a`,
where :math:`r` is the relative tolerance, and :math:`a` is the absolute tolerance. The tolerance
:math:`\theta` is clipped below at :math:`10^{-15}` to avoid rejection due to rounding errors.

:param actual: Actual value :math:`x`.
:param desired: Desired value :math:`y`.
:param rtol: Relative tolerance :math:`r`.
:param atol: Absolute tolerance :math:`a`.
:retval 1: If the values are close.
:retval 0: If the values are not close.
*/
int is_close(real actual, real desired, real rtol, real atol) {
    // We always allow a tolerance of at least 1e-15 in case there are rounding errors.
    real tol = fmax(atol + rtol * abs(desired), 1e-15);
    if (abs(actual - desired) <= tol) {
        return 1;
    }
    return 0;
}

/**
Assert that two values are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(real actual, real desired, real rtol, real atol) {
    if (!is_close(actual, desired, rtol, atol)) {
        reject(actual, " is not close to ", desired);
    }
}

/**
Assert that two values are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(real actual, real desired) {
    assert_close(actual, desired, 1e-6, 0);
}

/**
Check whether a possibly complex value is finite.

:param x: Value to check.
:retval 1: If the values are close.
:retval 0: If the values are not close.
*/
int is_finite(complex x) {
    real rx = get_real(x);
    real ix = get_imag(x);
    if (is_nan(rx) || is_nan(ix) || is_inf(rx) || is_inf(ix)) {
        return 0;
    }
    return 1;
}


// Vectors -----------------------------------------------------------------------------------------

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(vector actual, vector desired, real rtol, real atol) {
    int n = size(desired);
    int m = size(actual);
    if (m != n) {
        reject("number of elements are not equal: size(desired)=", n, "; size(actual)=", m);
    }
    for (i in 1:size(actual)) {
        if (!is_close(actual[i], desired[i], rtol, atol)) {
            reject(actual[i], " is not close to ", desired[i], " at position ", i);
        }
    }
}

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(vector actual, vector desired) {
    assert_close(actual, desired, 1e-6, 0);
}

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(vector actual, real desired, real rtol, real atol) {
    assert_close(actual, rep_vector(desired, size(actual)), rtol, atol);
}

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(vector actual, real desired) {
    assert_close(actual, desired, 1e-6, 0);
}

/**
Check whether all elements of a vector are finite.

:param x: Vector to check.
:retval 1: If the values are close.
:retval 0: If the values are not close.
*/
int is_finite(vector x) {
    for (i in 1:size(x)) {
        if(!is_finite(x[i])) {
            return 0;
        }
    }
    return 1;
}

/**
Assert that all elements of a vector are finite.

:param: Vector to check.
*/
void assert_finite(vector x) {
    int n = size(x);
    for (i in 1:n) {
        if (!is_finite(x[i])) {
            reject(x[i], " at index ", i, " is not finite");
        }
    }
}

// Matrices ----------------------------------------------------------------------------------------

/**
Pretty-print a matrix.
*/
void print_matrix(complex_matrix x) {
    print("matrix with ", rows(x), " rows and ", cols(x), " columns");
    for (i in 1:rows(x)) {
        print(x[i]);
    }
}

/**
Assert that two matrices are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(matrix actual, matrix desired, real rtol, real atol) {
    array [2] int nshape = dims(desired);
    array [2] int mshape = dims(actual);
    if (mshape[1] != nshape[1]) {
        reject("number of rows are not equal: dims(desired)[1]=", nshape[1], "; size(actual)[1]=",
               mshape[1]);
    }
    if (mshape[2] != nshape[2]) {
        reject("number of columns are not equal: dims(desired)[2]=", nshape[2],
               "; size(actual)[2]=", mshape[2]);
    }
    for (i in 1:nshape[1]) {
        for (j in 1:nshape[2]) {
            if (!is_close(actual[i, j], desired[i, j], rtol, atol)) {
                reject(actual[i, j], " is not close to ", desired[i, j], " at row ", i, ", column ",
                       j);
            }
        }
    }
}

/**
Assert that two matrices are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(matrix actual, matrix desired) {
    assert_close(actual, desired, 1e-6, 0);
}

/**
Assert that two matrices are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(matrix actual, real desired, real rtol, real atol) {
    array [2] int shape = dims(actual);
    assert_close(actual, rep_matrix(desired, shape[1], shape[2]), rtol, atol);
}

/**
Assert that two matrices are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(matrix actual, real desired) {
    assert_close(actual, desired, 1e-6, 0);
}

// Complex vectors ---------------------------------------------------------------------------------

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(complex_vector actual, complex_vector desired, real rtol, real atol) {
    int n = size(desired);
    int m = size(actual);
    if (m != n) {
        reject("number of elements are not equal: size(desired)=", n, "; size(actual)=", m);
    }
    assert_close(get_real(actual), get_real(desired), rtol, atol);
    assert_close(get_imag(actual), get_imag(desired), rtol, atol);
}


/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(complex_vector actual, complex_vector desired) {
    assert_close(actual, desired, 1e-6, 0);
}

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(complex_vector actual, complex desired, real rtol, real atol) {
    assert_close(actual, rep_vector(desired, size(actual)), rtol, atol);
}

/**
Assert that two vectors are close. See :cpp:func:`is_close` for description of parameters.
*/
void assert_close(complex_vector actual, complex desired) {
    assert_close(actual, desired, 1e-6, 0);
}


/**
Assert that all elements of a complex vector are finite.

:param: Vector to check.
*/
void assert_finite(complex_vector x) {
    int n = size(x);
    for (i in 1:n) {
        if (!is_finite(x[i])) {
            reject(x[i], " at index ", i, " is not finite");
        }
    }
}

// Shorthand for creating containers ---------------------------------------------------------------

vector zeros(int n) {
    return rep_vector(0, n);
}

vector ones(int n) {
    return rep_vector(0, n);
}

// Real Fourier transforms -------------------------------------------------------------------------

/**
Evaluate the complex conjugate.
*/
complex conjugate(complex x) {
    return get_real(x) - 1.0i * get_imag(x);
}

/**
Evaluate the complex conjugate.
*/
complex_vector conjugate(complex_vector x) {
    return get_real(x) - 1.0i * get_imag(x);
}

/**
Evaluate the complex conjugate.
*/
complex_row_vector conjugate(complex_row_vector x) {
    return get_real(x) - 1.0i * get_imag(x);
}

/**
Evaluate the complex conjugate.
*/
complex_matrix conjugate(complex_matrix x) {
    return get_real(x) - 1.0i * get_imag(x);
}

/**
Compute the one-dimensional discrete Fourier transform for real input.

:param y: Real signal with `n` elements to transform.
:returns: Truncated vector of Fourier coefficients with `n %/% 2 + 1` elements.
*/
complex_vector rfft(vector y) {
    return fft(y)[:size(y) %/% 2 + 1];
}

/**
Compute the one-dimensional inverse discrete Fourier transform for real output.

:param z: Truncated vector of Fourier coefficents with `n %/% 2 + 1` elements.
:param n: Length of the signal (required because the length of the signal cannot be determined from
    `z` alone).
:returns: Real signal with `n` elements.
*/
vector inv_rfft(complex_vector z, int n) {
    complex_vector[n] x;
    int nrfft = n %/% 2 + 1;
    int ncomplex = (n - 1) %/% 2;
    x[1:nrfft] = z[1:nrfft];
    x[nrfft + 1:n] = conjugate(reverse(z[2:1 + ncomplex]));
    return get_real(inv_fft(x));
}

/**
Compute the two-dimensional discrete Fourier transform for real input.

:param y: Real signal with `n` rows and `m` columns to transform.
:returns: Truncated vector of Fourier coefficients with `n` rows and `m %/% 2 + 1` elements.
*/
complex_matrix rfft2(matrix y) {
    return fft2(y)[:, :cols(y) %/% 2 + 1];
}

/**
Compute the two-dimensional inverse discrete Fourier transform for real output.

:param z: Truncated vector of Fourier coefficients with `n` rows and `m %/% 2 + 1` elements.
:param m: Number of columns of the signal (required because the number of columns cannot be
    determined from `z` alone).
:returns: Real signal with `n` rows and `m` columns.
*/
matrix inv_rfft2(complex_matrix z, int m) {
    int n = rows(z);
    complex_matrix[n, m] x;
    int mrfft = m %/% 2 + 1;
    int mcomplex = (m - 1) %/% 2;
    x[:, 1:mrfft] = z[:, 1:mrfft];
    // Fill redundant values.
    for (i in 1:n) {
        x[i, mrfft + 1:m] = conjugate(reverse(z[i, 2:1 + mcomplex]));
    }
    // Reverse the order to account for negative frequencies.
    for (i in mrfft + 1:mrfft + mcomplex) {
        x[2:, i] = reverse(x[2:, i]);
    }
    return get_real(inv_fft2(x));
}
