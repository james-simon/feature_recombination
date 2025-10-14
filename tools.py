import numpy as np
import torch

from kernels import krr
from utils import ensure_numpy, ensure_torch



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

## same as fine_beta in essence
def estimate_beta(K, y, n_trains, n_test=5_000, ridge=1e-3, n_trials=None):

    n_trains = np.asarray(n_trains)
    if n_trials is None:
        try:
            n_trials = trial_count_fn(int(n_trains.max()))
        except NameError:
            n_trials = 10

    beta, intercept, sizes, test_mses = find_beta(K, y, num_estimators=len(n_trains), n_test=n_test,
                                                  n_trials=int(n_trials), sizes=n_trains, ridge=ridge)

    coeff = 10.0 ** float(intercept)
    return float(beta), coeff, test_mses

def trial_count_fn(n):
    if n <= 50:
        return 20
    elif n <= 500:
        return 10
    elif n <= 5000:
        return 3
    else:
        return 1
    
def grf(H, y, chunk_size=10_000, idxs=None):
    _, P = H.shape
    if idxs is None:
        idxs = np.arange(P)
    vhat = ensure_torch(np.zeros(P))
    H_norm = ensure_torch(np.zeros(P))
    residual = ensure_torch(y, clone=True)
    residual /= torch.linalg.norm(residual)
    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        idx_chunk = idxs[start:end]
        H_chunk = ensure_torch(H[:, idx_chunk])
        H_norm[idx_chunk] = torch.linalg.norm(H_chunk, axis=0)
        for i, t in enumerate(idx_chunk):
            h_t = H_chunk[:, i]
            vhat[t] = torch.dot(h_t, residual) / H_norm[t]
            residual -= vhat[t] * h_t / H_norm[t]
        del H_chunk
        torch.cuda.empty_cache()
    vhat, residual, H_norm = map(ensure_numpy, (vhat, residual, H_norm))
    return vhat, residual, H_norm