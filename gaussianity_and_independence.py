import numpy as np
from scipy.stats import norm
import torch
from utils import ensure_numpy, ensure_torch 

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

def full_analysis(X, kerneltype, kernel_width, top_hea_eigmode=3000):
    X_gaussian = gaussianize_marginals(X)
    X_independent = independentize_data(X, bsz=X.shape[0])
    X_gaussian_independent = gaussianize_data(X)
    
    outdict = {}

    def get_everything(X_in, kerneltype, kernel_width, top_hea_eigmode):
        torch.cuda.empty_cache()
        from notebook_fns import get_standard_tools
        monomials, kernel, H, hea_eigvals, data_eigvals = get_standard_tools(X_in, kerneltype, kernel_width, top_mode_idx=top_hea_eigmode)

        eigvals, eigvecs = kernel.eigendecomp()
        pdf, cdf, quartiles = kernel.kernel_function_projection(H)
        return {"monomials": monomials, "kernel": kernel, "kernel": H, "eigvals": eigvals, "pdf": pdf, "cdf": cdf,
                "quartiles": quartiles, "hea_eigvals": hea_eigvals, "data_eigvals": data_eigvals}
    
    outdict["Normal"] = get_everything(X, kerneltype, kernel_width, top_hea_eigmode)
    outdict["Gaussian"] = get_everything(X_gaussian, kerneltype, kernel_width, top_hea_eigmode)
    outdict["Independent"] = get_everything(X_independent, kerneltype, kernel_width, top_hea_eigmode)
    outdict["Gaussian Independent"] = get_everything(X_gaussian_independent, kerneltype, kernel_width, top_hea_eigmode)
    return outdict

def eigvecs_to_independent(eigenvectors, bsz=None, rng = None, to_torch=True):
    rng = np.random.default_rng() if rng is None else rng
    n, dim = eigenvectors.shape #n = num datapoints
    bsz = n if bsz is None else bsz
    independent_components = torch.zeros((bsz, dim))

    for i in range(bsz):
        # For each column, randomly select one value from the n rows
        random_rows = rng.integers(0, n, size=dim)
        independent_components[i] = eigenvectors[random_rows, np.arange(dim)]

    return ensure_torch(independent_components) if to_torch else independent_components

def independentize_data(X, bsz=1, rng=None):
    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
    eigenvectors = U @ torch.diag(S)

    independent_data = eigvecs_to_independent(eigenvectors, bsz, rng)
    return independent_data @ Vt