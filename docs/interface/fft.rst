Likelihood Evaluations Based on Fast Fourier Transforms
=======================================================

The module implements :ref:`fourier-methods` for `one-dimensional signals`_ and `two-dimensional signals`_. The functions of immediate use to practitioners are :stan:func:`gp_rfft_lpdf` to evaluate the likelihood of a one-dimensional signal and :stan:func:`gp_inv_rfft` to construct a :ref:`non-centered parameterization <parameterizations>` using white noise. The two-dimensional analogues are :stan:func:`gp_rfft2_lpdf` and :stan:func:`gp_inv_rfft2`. The module also provides functions to evaluate kernels directly in the Fourier domain [#]_, including :stan:func:`gp_periodic_exp_quad_cov_rfft` and :stan:func:`gp_periodic_matern_cov_rfft`.

Utility Functions
-----------------

The following functions, and their two-dimensional analogues, primarily exist as utility functions but may be useful for more complex models.

- :stan:func:`gp_rfft` transforms Gaussian process realizations to the Fourier domain and scales the coefficients such that they are white noise under the Gaussian process prior.
- :stan:func:`gp_rfft_log_abs_det_jac` evaluates the log absolute determinant of the Jacobian associated with the transformations :stan:func:`gp_rfft`.

Together, these two functions are used by :stan:func:`gp_rfft_lpdf` to evaluate the likelihood.

- :stan:func:`gp_unpack_rfft` unpacks complex Fourier coefficients of size :code:`n %/% 2 + 1` to a real vector of size :code:`n` for easier manipulation.
- :stan:func:`gp_pack_rfft` is, unsurprisingly, the inverse of :stan:func:`gp_unpack_rfft` and packs a real vector of size :code:`n` into a complex vector of size :code:`n %/% 2 + 1` such that the inverse RFFT can be applied.
- :stan:func:`gp_evaluate_rfft_scale` evaluates the expected standard deviation of Fourier coefficients obtained by transforming a Gaussian process. The values are arranged to match the output of :stan:func:`gp_unpack_rfft`.

Function Reference
------------------

One-dimensional Signals
^^^^^^^^^^^^^^^^^^^^^^^

.. stan:autodoc:: ../../stan/gptools/fft1.stan

Two-dimensional Signals
^^^^^^^^^^^^^^^^^^^^^^^

.. stan:autodoc:: ../../stan/gptools/fft2.stan

.. [#] Fourier-domain kernels are implemented by discretizing their power spectrum naively. This approach works well if the number of grid points is large and the correlation length is small compared with the size of the domain. More `sophisticated techniques <https://proceedings.neurips.cc/paper/2020/file/92bf5e6240737e0326ea59846a83e076-Paper.pdf>`_ may be required otherwise.
