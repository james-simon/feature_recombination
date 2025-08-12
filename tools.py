import numpy as np
import torch

from feature_decomp import generate_fra_monomials
from data import get_matrix_hermites
from kernels import krr
from utils import ensure_torch


def get_standard_tools(X, kerneltype, kernel_width, top_mode_idx=3000, data_eigvals=None, kmax=20):
    
    if data_eigvals is None:
        N, _ = X.shape
        S = torch.linalg.svdvals(X)
        data_eigvals = S**2 / (S**2).sum()

    kernel = kerneltype(X, kernel_width=kernel_width)
    eval_level_coeff = kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff, kmax=kmax)
    H = ensure_torch(get_matrix_hermites(X, monomials))

    return monomials, kernel, H, fra_eigvals, data_eigvals

def get_test_mses(K, y, num_estimators=20, n_test=100, n_trials=20, **kwargs):
    sizes = np.logspace(np.log10(kwargs.get("size_start", 100)), np.log10(K.shape[0]-n_test)-kwargs.get('size_offset', 0.01), num=num_estimators)
    K = ensure_torch(K)
    y = ensure_torch(y)
    test_mses = np.zeros((n_trials, num_estimators))
    for i, n in enumerate(sizes):
        n = int(n)
        for trial in range(n_trials):
            train_idx = torch.arange(0, n)
            test_idx   = torch.arange(K.shape[0] - n_test, K.shape[0])
            idxs       = torch.concatenate([train_idx, test_idx]).to(K.device)
            K_sub, y_sub = K[idxs[:, None], idxs[None, :]], y[idxs]
            (y_hat_test, y_test), _ = krr(K_sub, y_sub, n_train=n, n_test=n_test, ridge=kwargs.get("ridge", 1e-20))
            
            test_mse = ((y_test - y_hat_test) ** 2).mean(axis=0)
            test_mse = test_mse.sum().item()

            test_mses[trial, i] = test_mse
            torch.cuda.empty_cache()
    return sizes, test_mses

def find_beta(K, y, num_estimators=20, n_test=100, n_trials=20, **kwargs):
    sizes, test_mses = get_test_mses(K=K, y=y, num_estimators=num_estimators, n_test=n_test, n_trials=n_trials, **kwargs)

    log_sizes = torch.log10(ensure_torch(sizes))
    log_sizes_centered_w_intercept = torch.column_stack((log_sizes, torch.ones_like(log_sizes)))

    sol = torch.linalg.lstsq(log_sizes_centered_w_intercept, ensure_torch(torch.log10(ensure_torch(test_mses).mean(axis=0))).unsqueeze(1)).solution.squeeze()
    slope          = float(sol[0])
    intercept     = float(sol[1])
    beta = -slope+1

    return beta, intercept, sizes, test_mses