import numpy as np
import torch
import math
from feature_decomp import generate_fra_monomials
from data import get_matrix_hermites


def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ensure_torch(x, dtype=torch.float32):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype, device=DEVICE)
    return x.to(dtype=dtype, device=DEVICE)


def get_data_eigvals(X):
    N, _ = X.shape
    S = torch.linalg.svdvals(X)
    # to make norm(x)~1 on average (f)
    X *= torch.sqrt(N / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()
    return data_eigvals

def get_standard_tools(X, kerneltype, kernel_width, top_mode_idx = 3000, data_eigvals = None, kmax=20):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if data_eigvals is None:
        data_eigvals = get_data_eigvals(X)

    kernel = kerneltype(X, kernel_width=kernel_width)
    eval_level_coeff = kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff, kmax=kmax)
    H = get_matrix_hermites(X, monomials).to(DEVICE)

    return monomials, kernel, H, fra_eigvals, data_eigvals

def grab_eigval_distributions(X):
    # N, d = X.shape

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    #each column is an vector corresponding to the i-th eigenvalue
    eigenvector_array = U @ torch.diag(S)# * np.sqrt(N)

    return eigenvector_array, Vt

def cossim_per_eigvec(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    return np.sum(v1_norm * v2_norm, axis=1)

#helper fns for experimental tests
def find_iterables(d, no_list=["H", "y"]):
    return {k: v for k, v in d.items() if (isinstance(v, (list, np.ndarray)) and k not in no_list)}

def find_statics(d):
    return {k: v for k, v in d.items() if not isinstance(v, (list, np.ndarray))}