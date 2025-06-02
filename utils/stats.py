import torch
import numpy as np
from utils import get_matrix_hermites
from feature_decomp import generate_fra_monomials

def get_data_eigvals(X):
    N, _ = X.shape
    S = torch.linalg.svdvals(X)
    # to make norm(x)~1 on average (f)
    X *= torch.sqrt(N / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()
    return data_eigvals

def get_standard_tools(X, kerneltype, kernel_width, top_mode_idx = 3000, data_eigvals = None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if data_eigvals is None:
        data_eigvals = get_data_eigvals(X)

    kernel = kerneltype(X, kernel_width=kernel_width)
    eval_level_coeff = kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff)
    H = get_matrix_hermites(X, monomials).to(DEVICE)

    return monomials, kernel, H, fra_eigvals, data_eigvals

def grab_eigval_distributions(X):
    # N, d = X.shape

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    #each column is an vector corresponding to the i-th eigenvalue
    eigenvector_array = U @ torch.diag(S)# * np.sqrt(N)

    return eigenvector_array, Vt