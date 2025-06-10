import numpy as np
import torch
import math
from feature_decomp import generate_fra_monomials

def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ensure_torch(x, dtype=torch.float32):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype).to(DEVICE)
    return x.to(dtype).to(DEVICE)


def get_matrix_hermites(X, monomials):
    N, _ = X.shape
    U, _, _ = torch.linalg.svd(X, full_matrices=False)
    X_norm = np.sqrt(N) * U

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
    }

    H = ensure_torch(torch.zeros((N, len(monomials))))
    for i, monomial in enumerate(monomials):
        h = ensure_torch(torch.ones(N) / np.sqrt(N))
        for d_i, exp in monomial.items():
            Z = np.sqrt(math.factorial(exp))
            h *= hermites[exp](X_norm[:, d_i]) / Z
        H[:, i] = h
    return H


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