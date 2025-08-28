import numpy as np
import torch
from tqdm import tqdm, trange

from ExptTrace import ExptTrace

from feature_decomp import generate_fra_monomials
from data import get_matrix_hermites
from kernels import krr
from utils import ensure_numpy, ensure_torch


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
    sizes = kwargs.get('sizes', None)
    if sizes is None:
        sizes = np.logspace(np.log10(kwargs.get("size_start", 100)), np.log10(K.shape[0]-n_test)-kwargs.get('size_offset', 0.01), num=num_estimators)
    K = ensure_torch(K)
    y = ensure_torch(y)
    test_mses = np.zeros((n_trials, len(sizes)))
    for i, n in enumerate(sizes):
        n = int(n)
        for trial in range(n_trials):
            (y_hat_test, y_test), _ = krr(K, y, n_train=n, n_test=n_test, ridge=kwargs.get("ridge", 1e-20))
            
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


def estimate_beta(K, y, n_trains, n_test=5_000, n_tailstart=800):
    
    def get_ntrials(ntrain):
        if ntrain < 100: return 20
        elif ntrain < 1000: return 10
        elif ntrain < 10000: return 5
        else: return 1
    
    assert n_trains.dtype == int
    assert n_trains.min() > 0
    assert n_trains.max() + n_test <= K.shape[0]
    K = ensure_torch(K)
    y = ensure_torch(y)
    var_axes = ["trial", "ntrain"]
    et_yhat = ExptTrace(var_axes)
    for n_train in tqdm(n_trains):
        for trial in range(get_ntrials(n_train)):
            (y_hat, _), _ = krr(K, y, n_train, n_test=n_test, ridge=1e-3)
            et_yhat[trial, n_train] = y_hat.cpu().numpy()
        torch.cuda.empty_cache()

    yhat = et_yhat[:, :]
    ystar = ensure_numpy(y[-n_test:])
    mse_trials = ((yhat - ystar)**2).mean(axis=-1)
    mse = mse_trials.mean(axis=0)
    
    # Fit powerlaw to the tail of the curve (n >= n_tailstart)
    tail_mask = n_trains >= n_tailstart
    assert tail_mask.sum() >= 2
    log_n = np.log(n_trains[tail_mask])
    log_mse = np.log(mse[tail_mask])
    poly = np.polynomial.Polynomial.fit(log_n, log_mse, deg=1)
    intercept, slope = poly.convert().coef
    coeff, beta = np.e**intercept, -slope + 1
    return beta, coeff, mse_trials


def grf(H, y, P):
    if P is None:
        P = H.shape[1]
    assert P <= H.shape[1], "P must not exceed num modes"

    H = ensure_torch(H)
    y = ensure_torch(y)
    vhat = ensure_torch(torch.zeros(P))
    uncaptured = np.zeros(P)
    residual = y.clone()
    with trange(P, desc="GRF", unit="step", total=P) as pbar:
        for j in pbar:
            phi_j = H[:, j]
            vhat[j] = torch.dot(phi_j, residual) / torch.linalg.norm(phi_j) ** 2
            residual -= vhat[j] * phi_j
            uncaptured[j] = residual.var().item()
            pbar.set_postfix(uncaptured=uncaptured[j])
    return ensure_numpy(vhat), uncaptured
