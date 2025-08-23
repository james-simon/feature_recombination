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


def get_matrix_hermites(X, monomials, previously_normalized=False):
    N, _ = X.shape
    if not previously_normalized:
        U, _, _ = torch.linalg.svd(X, full_matrices=False)
        X_norm = np.sqrt(N) * U
    else:
        X_norm = X

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
        13: lambda x: x**13 - 78*x**11 + 2145*x**9 - 25740*x**7 + 135135*x**5 - 270270*x**3 + 135135*x,
        14: lambda x: x**14 - 91*x**12 + 3003*x**10 - 45045*x**8 + 315315*x**6 - 945945*x**4 + 945945*x**2 - 135135,
        15: lambda x: x**15 - 105*x**13 + 4095*x**11 - 76545*x**9 + 675675*x**7 - 2702700*x**5 + 4729725*x**3 - 2027025*x,
        16: lambda x: x**16 - 120*x**14 + 5460*x**12 - 120120*x**10 + 1351350*x**8 - 7567560*x**6 + 18918900*x**4 - 20270250*x**2 + 34459425,
        17: lambda x: x**17 - 136*x**15 + 7140*x**13 - 180180*x**11 + 2297295*x**9 - 15315300*x**7 + 51081030*x**5 - 87513450*x**3 + 34459425*x,
        18: lambda x: x**18 - 153*x**16 + 9180*x**14 - 257400*x**12 + 3783780*x**10 - 30630600*x**8 + 122522400*x**6 - 229729500*x**4 + 172972500*x**2 - 34459425,
        19: lambda x: x**19 - 171*x**17 + 11628*x**15 - 375375*x**13 + 6432420*x**11 - 61261200*x**9 + 306306000*x**7 - 765765000*x**5 + 875134500*x**3 - 310134825*x,
        20: lambda x: x**20 - 190*x**18 + 14250*x**16 - 513513*x**14 + 10210200*x**12 - 117117000*x**10 + 765765000*x**8 - 2677114440*x**6 + 4670678100*x**4 - 3101348250*x**2 + 654729075,
    }

    H = ensure_torch(torch.zeros((N, len(monomials))))
    for i, monomial in enumerate(monomials):
        h = ensure_torch(torch.ones(N) / np.sqrt(N))
        for d_i, exp in monomial.items():
            Z = np.sqrt(math.factorial(exp))
            h *= hermites[exp](X_norm[:, d_i]) / Z
        H[:, i] = h
    return H


def get_powerlaw_target(H, source_exp, offset=6, normalizeH=True, include_noise=False):
    if source_exp <= 1:
        raise ValueError("source_exp must be > 1 for powerlaw target")
    if offset < 1:
        raise ValueError("offset ≥ 1required")
    N, P = H.shape
    if normalizeH:
        H /= torch.linalg.norm(H, dim=0, keepdim=True)
    squared_coeffs = get_powerlaw(P, source_exp, offset=offset)
    # Generate random signs for coefficients
    signs = -1 + 2*ensure_torch(torch.randint(0, 2, size=squared_coeffs.shape))
    coeffs = torch.sqrt(squared_coeffs) * signs.float()
    y = H @ coeffs
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