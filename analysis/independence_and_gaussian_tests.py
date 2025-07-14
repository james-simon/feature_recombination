import torch
from .gaussianity_tests import gaussianize_marginals, gaussianize_data
from .independence_tests import independentize_data
from tools import get_standard_tools

def full_analysis(X, kerneltype, kernel_width, top_fra_eigmode=3000):
    X_gaussian = gaussianize_marginals(X)
    X_independent = independentize_data(X, bsz=X.shape[0])
    X_gaussian_independent = gaussianize_data(X)
    # X_independent_gaussian = independentize_data(X_gaussian, bsz=X.shape[0])

    outdict = {}

    def get_everything(X_in, kerneltype, kernel_width, top_fra_eigmode):
        torch.cuda.empty_cache()
        monomials, kernel, H, fra_eigvals, data_eigvals = get_standard_tools(X_in, kerneltype, kernel_width, top_mode_idx=top_fra_eigmode)

        eigvals, eigvecs = kernel.eigendecomp()
        pdf, cdf, quartiles = kernel.kernel_function_projection(H)
        return {"monomials": monomials, "kernel": kernel, "kernel": H, "eigvals": eigvals, "pdf": pdf, "cdf": cdf,
                "quartiles": quartiles, "fra_eigvals": fra_eigvals, "data_eigvals": data_eigvals}
    
    outdict["Normal"] = get_everything(X, kerneltype, kernel_width, top_fra_eigmode)
    outdict["Gaussian"] = get_everything(X_gaussian, kerneltype, kernel_width, top_fra_eigmode)
    outdict["Independent"] = get_everything(X_independent, kerneltype, kernel_width, top_fra_eigmode)
    outdict["Gaussian Independent"] = get_everything(X_gaussian_independent, kerneltype, kernel_width, top_fra_eigmode)
    return outdict