import numpy as np
from scipy.stats import norm
import torch
from utils import ensure_numpy, ensure_torch 
from .independence_tests import independentize_data

def get_emperical_pdf(X, num_bins=100, tol=1e-3):
    counts, bin_edges = np.histogram(X, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    pdf = counts * bin_width
    assert np.abs(np.cumsum(pdf)[-1]-1) <= tol, f"PDF estimator is outside of tolderance {tol} away from area 1"
    return bin_centers, pdf

def get_emperical_cdf(X):
    ranks = np.argsort(np.argsort(X, axis=0), axis=0)
    uniform = (ranks + 1) / (X.shape[0] + 1)
    return uniform #uniform distribution while preserving the relative rank

# def cdf_to_gaussian(uniform_cdf, S):
#     gaussian_samples = norm.ppf(uniform_cdf) * S
#     return gaussian_samples

# def sample_from_cdf(cdf, num_samples=10000):
#     uniform_samples = np.random.rand(num_samples)

#     bin_indices = np.searchsorted(cdf, uniform_samples)

#     bin_left_edges = bin_edges[bin_indices]
#     bin_right_edges = bin_edges[bin_indices + 1]
#     random_within_bin = np.random.rand(num_samples)

#     samples = bin_left_edges + random_within_bin * (bin_right_edges - bin_left_edges)
#     return samples

def eigvecs_to_gaussian(X, S=None, to_torch=True):
    if S is None:
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
    X = ensure_numpy(X)
    S = ensure_numpy(S)
    uniform = get_emperical_cdf(X)
    gaussian_data = norm.ppf(uniform)

    if S is not None:
        S = np.asarray(S)
        if S.ndim == 0:
            gaussian_data *= S
        else:
            gaussian_data *= S.reshape(1, -1)  # broadcast to columns
    return ensure_torch(gaussian_data) if to_torch else gaussian_data

def gaussianize_marginals(X):
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    eigenvectors = U @ torch.diag(S)
    gaussian_data = eigvecs_to_gaussian(eigenvectors, S)
    return gaussian_data @ Vt

def gaussianize_data(X):
    _, _, Vt = torch.linalg.svd(X, full_matrices=False)
    X_independent = independentize_data(X, bsz=X.shape[0])
    gaussianized_data = gaussianize_marginals(X_independent)
    return gaussianized_data @ Vt

def percent_gaussian(X, alpha=0.5, S=None):
    modified_dist = (1-alpha) * X + alpha * gaussianize_data(X)
    return modified_dist