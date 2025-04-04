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
    X_rot, PCA_dirs, data_covar_eigvals = PCA(X)
    X_norm = X_rot / torch.sqrt(data_covar_eigvals[None, :])
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
        11: lambda x: x**11 - 55*x**9 + 990*x**7 - 6930*x**5 + 17325*x**3 - 10395*x,
        12: lambda x: x**12 - 66*x**10 + 1485*x**8 - 13860*x**6 + 51975*x**4 - 62370*x**2 + 10395,
        13: lambda x: x**13 - 78*x**11 + 2145*x**9 - 25080*x**7 + 135135*x**5 - 270270*x**3 + 135135*x,
        14: lambda x: x**14 - 91*x**12 + 3003*x**10 - 40040*x**8 + 240240*x**6 - 675675*x**4 + 945945*x**2 - 135135,
        15: lambda x: x**15 - 105*x**13 + 4095*x**11 - 60060*x**9 + 450450*x**7 - 1621620*x**5 + 2837835*x**3 - 2027025*x,
    }


    H = torch.zeros((N, len(monomials)))
    for i, monomial in enumerate(monomials):
        h = torch.ones(N) / np.sqrt(N)
        for d_i, exp in monomial.items():
            Z = torch.tensor(np.sqrt(math.factorial(exp)))
            h *= hermites[exp](X_norm[:, d_i]).to("cpu") / Z
        H[:, i] = h
    return H