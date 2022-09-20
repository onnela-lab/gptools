// Scalars -----------------------------------------------------------------------------------------
void assert_equal(int actual, int desired) {
    if (actual != desired) {
        reject(actual, " is not equal to ", desired);
    }
}

int is_close(real actual, real desired, real rtol, real atol) {
    // We always allow a tolerance of at least 1e-15 in case there are rounding errors.
    real tol = fmax(atol + rtol * abs(desired), 1e-15);
    if (abs(actual - desired) <= tol) {
        return 1;
    }
    return 0;
}

void assert_close(real actual, real desired, real rtol, real atol) {
    if (!is_close(actual, desired, rtol, atol)) {
        reject(actual, " is not close to ", desired);
    }
}

void assert_close(real actual, real desired) {
    assert_close(actual, desired, 1e-6, 0);
}

// Vectors -----------------------------------------------------------------------------------------

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

void assert_close(vector actual, vector desired) {
    assert_close(actual, desired, 1e-6, 0);
}

void assert_close(vector actual, real desired, real rtol, real atol) {
    assert_close(actual, rep_vector(desired, size(actual)), rtol, atol);
}

void assert_close(vector actual, real desired) {
    assert_close(actual, desired, 1e-6, 0);
}

// Matrices ----------------------------------------------------------------------------------------

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

void assert_close(matrix actual, matrix desired) {
    assert_close(actual, desired, 1e-6, 0);
}

void assert_close(matrix actual, real desired, real rtol, real atol) {
    array [2] int shape = dims(actual);
    assert_close(actual, rep_matrix(desired, shape[1], shape[2]), rtol, atol);
}

void assert_close(matrix actual, real desired) {
    assert_close(actual, desired, 1e-6, 0);
}
