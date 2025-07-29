import numpy as np
import torch as torch   
import math
from scipy.special import zeta

from utils import ensure_torch


def get_powerlaw(P, exp, offset=3, normalize=True):
    pl = ensure_torch((offset+np.arange(P)) ** -exp)
    if normalize:
        pl /= pl.sum()
    return pl


def get_matrix_hermites(X, monomials):
    N, _ = X.shape
    U, _, _ = torch.linalg.svd(X, full_matrices=False)
    X_norm = np.sqrt(N) * U

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

    H = ensure_torch(torch.zeros((N, len(monomials))))
    for i, monomial in enumerate(monomials):
        h = ensure_torch(torch.ones(N) / np.sqrt(N))
        for d_i, exp in monomial.items():
            Z = np.sqrt(math.factorial(exp))
            h *= hermites[exp](X_norm[:, d_i]) / Z
        H[:, i] = h
    return H


def get_powerlaw_target(H, source_exp, offset=6, include_noise=True):
    if source_exp <= 1:
        raise ValueError("source_exp must be > 1 for powerlaw target")
    if offset < 1:
        raise ValueError("offset ≥ 1required")
    N, P = H.shape
    squared_coeffs = get_powerlaw(P, source_exp, offset=offset)
    y = H @ torch.sqrt(squared_coeffs)
    if include_noise:
        totalsum = zeta(source_exp, offset)  # sum_{k=offset  }^infty k^{-exp}
        tailsum = zeta(source_exp, offset+P) # sum_{k=offset+P}^infty k^{-exp}
        noise_var = tailsum/(totalsum - tailsum)
        noise = torch.normal(0, np.sqrt(noise_var / N), y.shape)
        # snr = y @ y / (noise @ noise)
        y /= torch.linalg.norm(y)
        y += ensure_torch(noise)
    # we expect size(y_i) ~ 1
    y = np.sqrt(N) * y / torch.linalg.norm(y)    
    return y