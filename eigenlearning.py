import numpy as np
from scipy.optimize import bisect
from utils import ensure_numpy


def compute_learnabilities(n, eigvals, ridge=0):
    
    def _compute_lrn(n, eigvals, ridge):
        eigvals = ensure_numpy(eigvals)
        conservation_law = lambda kap: (eigvals/(eigvals+kap)).sum() + ridge/kap - n
        kappa = bisect(conservation_law, 1e-25, 1e10, maxiter=128)
        learnabilities = eigvals / (eigvals + kappa)
        return learnabilities, kappa
    
    n = ensure_numpy(n)
    if np.ndim(n) == 0:
        return _compute_lrn(float(n), eigvals, ridge)
    if np.ndim(n) == 1:
        learnabilities = np.zeros((len(n), len(eigvals)))
        kappas = np.zeros(len(n))
        for i in range(len(n)):
            learnabilities[i], kappas[i] = _compute_lrn(n[i], eigvals, ridge)
        return learnabilities, kappas
    else:
        raise ValueError("n must be a scalar or 1D array/tensor")


def learning_curve(n, learnabilities, eigcoeffs, noise_var=0):
    n = ensure_numpy(n)
    if len(eigcoeffs) < learnabilities.shape[-1]:
        learnabilities = learnabilities[..., :len(eigcoeffs)]
    sum_axis = None if learnabilities.ndim == 1 else 1
    e0 = n / (n - (learnabilities**2).sum(axis=sum_axis))
    test_mse = e0 * (((1-learnabilities)**2 * eigcoeffs**2).sum(axis=sum_axis) + noise_var)
    return test_mse


def eigenlearning(n, eigvals, eigcoeffs, ridge=0, noise_var=0):
    n = ensure_numpy(n)
    learnabilities, kappa = compute_learnabilities(n, eigvals, ridge)
    test_mse = learning_curve(n, learnabilities, eigcoeffs, noise_var)
    train_mse = (ridge / (n * kappa))**2 * test_mse
    L = (eigcoeffs**2 * learnabilities).sum() / (eigcoeffs**2).sum()

    return {
        "kappa": kappa,
        "learnabilities": learnabilities,
        "target_learnability": L,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
