import numpy as np
import torch

from feature_decomp import generate_fra_monomials
from data import get_matrix_hermites
from kernels import krr
from utils import ensure_torch, get_data_eigvals

def get_standard_tools(X, kerneltype, kernel_width, top_mode_idx=3000, data_eigvals=None, kmax=20):
    
    if data_eigvals is None:
        data_eigvals = get_data_eigvals(X)

    kernel = kerneltype(X, kernel_width=kernel_width)
    eval_level_coeff = kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff, kmax=kmax)
    H = ensure_torch(get_matrix_hermites(X, monomials))

    return monomials, kernel, H, fra_eigvals, data_eigvals

def find_beta(K, y, num_estimators=20, n_test=100, n_trials=20, rng=np.random.default_rng(42)):
    sizes = np.logspace(0, np.log10(K.shape[0]-n_test)-0.2, num=num_estimators)
    K = ensure_torch(K)
    y = ensure_torch(y)
    test_mses = np.zeros((n_trials, num_estimators))
    for i, n in enumerate(sizes):
        n = int(n)
        for trial in range(n_trials):
            idxs = rng.choice(K.shape[0], size=n+n_test, replace=False)
            K_sub, y_sub = K[idxs[:, None], idxs[None, :]], y[idxs]
            _, test_mse, _ = krr(K_sub, y_sub, n_train=n, n_test=n_test, ridge=1e-20)
            test_mses[trial, i] = test_mse
            torch.cuda.empty_cache()
        
    sizes_w_intercept = torch.stack([ensure_torch(np.log10(sizes)), torch.ones_like(ensure_torch(sizes))], dim=1)   # [M, 2]  (M ≤ D after masking)
    sol = torch.linalg.lstsq(sizes_w_intercept, ensure_torch(np.log10(test_mses.mean(axis=0))).unsqueeze(1)).solution.squeeze()
    slope, intercept = sol[0], sol[1]
    beta = -slope+1

    return beta, intercept