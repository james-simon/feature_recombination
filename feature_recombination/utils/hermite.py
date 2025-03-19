import scipy as sp
import numpy as np

def hermite(k, x):
    """Compute the k-th probabilist's Hermite polynomial at x, normalized to have unit norm against the standard normal distribution."""

    return sp.special.hermite(k)(x / 2 ** .5) * 2 ** (-k / 2) / sp.special.factorial(k) ** .5

def hermite_product(X, exponents):
  n, d = X.shape

  fn_vals = np.ones(n)
  for idx in exponents:
    fn_vals *= hermite(exponents[idx], X[:, idx])

  return fn_vals