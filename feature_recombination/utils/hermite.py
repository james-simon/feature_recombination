import scipy as sp
import numpy as np
import torch
import math

from .general import PCA, ensure_torch, ensure_numpy

def hermite(k, x):
    """Compute the k-th probabilist's Hermite polynomial at x, normalized to have unit norm against the standard normal distribution."""

    return sp.special.hermite(k)(x / 2 ** .5) * 2 ** (-k / 2) / sp.special.factorial(k) ** .5

def hermite_product(X, exponents):
  n, d = X.shape

  fn_vals = np.ones(n)
  for idx in exponents:
    fn_vals *= hermite(exponents[idx], X[:, idx])

  return fn_vals

def get_matrix_hermites(X, monomials):
    X_rot, _, data_covar_eigvals = PCA(X)
    X_norm = X_rot / np.sqrt(data_covar_eigvals[None, :])
    N, _ = X.shape
    hermites = {
        1: lambda x: x,
        2: lambda x: x**2 - 1,
        3: lambda x: x**3 - 3*x,
        4: lambda x: x**4 - 6*x**2 + 3,
        5: lambda x: x**5 - 10*x**3 + 15*x,
        6: lambda x: x**6 - 15*x**4 + 45*x**2 - 15,
        7: lambda x: x**7 - 21*x**5 + 105*x**3 - 105*x,
        8: lambda x: x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105,
        9: lambda x: x**9 - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x,
        10: lambda x: x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 - 945,
    }

    H = torch.zeros((N, len(monomials)))
    for i, monomial in enumerate(monomials):
        h = ensure_torch(torch.ones(N) / np.sqrt(N))
        for d_i, exp in monomial.items():
            Z = torch.tensor(np.sqrt(math.factorial(exp)))
            h *= hermites[exp](X_norm[:, d_i]) / Z
        H[:, i] = h
    return H