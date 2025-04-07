import torch
import numpy as np
from .hermite import generate_fra_monomials, get_matrix_hermites

def get_structure(X, kerneltype, bandwidth, top_mode_idx = 3000, data_eigvals = None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, _ = X.shape
    if data_eigvals is None:
        S = torch.linalg.svdvals(X)
        # to make norm(x)~1 on average (f)
        X *= torch.sqrt(N / (S**2).sum())
        data_eigvals = S**2 / (S**2).sum()

    kernel = kerneltype(X, bandwidth=bandwidth)
    eval_level_coeff = kerneltype.get_level_coeff_fn(bandwidth=bandwidth, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff)
    H = get_matrix_hermites(X, monomials).to(DEVICE)

    return monomials, kernel, H, fra_eigvals, data_eigvals