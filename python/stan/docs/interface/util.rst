Utility Functions
=================

Utility functions support the implementation of :doc:`graph` and :doc:`fft`, such as

- :stan:func:`rfft`, :stan:func:`inv_rfft`, :stan:func:`rfft2`, and :stan:func:`inv_rfft2` to evaluate real fast Fourier transforms in one and two dimensions.
- :stan:func:`gp_conditional_loc_scale` to evaluate the conditional location and scale parameter of a univariate normal random variable given observations of correlated variables.
- functions to compare values and assertions for debugging.

Function Reference
------------------

.. stan:autodoc:: ../../gptools/stan/gptools/util.stan
