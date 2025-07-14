from feature_decomp import generate_fra_monomials
from data import get_matrix_hermites

from utils import ensure_torch, get_data_eigvals

def get_standard_tools(X, kerneltype, kernel_width, top_mode_idx=3000, data_eigvals=None, kmax=20):
    
    if data_eigvals is None:
        data_eigvals = get_data_eigvals(X)

    kernel = kerneltype(X, kernel_width=kernel_width)
    eval_level_coeff = kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff, kmax=kmax)
    H = ensure_torch(get_matrix_hermites(X, monomials))

    return monomials, kernel, H, fra_eigvals, data_eigvals