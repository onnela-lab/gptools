# gptoolsStan 0.1.0

Initial release.

# gptoolsStan 0.2.0

- Graph-based approximations of Gaussian processes now support different length scales along different dimensions (ported from https://github.com/onnela-lab/gptools/pull/23). This change requires cmdstan 2.36.0 or above due to a bug in `gp_matern32_cov` (see https://github.com/stan-dev/math/pull/3084).
- Functions ending `_log_abs_det_jacobian` have been renamed to `_log_abs_det_jac` to comply with upcoming changes (see https://github.com/stan-dev/stanc3/issues/1470 for details).
