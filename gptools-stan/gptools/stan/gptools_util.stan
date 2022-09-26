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
Check whether a value is finite.

:param x: Value to check.
:retval 1: If the values are close.
:retval 0: If the values are not close.
*/
int is_finite(real x) {
    if (is_nan(x) || is_inf(x)) {
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

// Matrices ----------------------------------------------------------------------------------------

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

// Shorthand for creating containers ---------------------------------------------------------------

vector zeros(int n) {
    return rep_vector(0, n);
}

vector ones(int n) {
    return rep_vector(0, n);
}
