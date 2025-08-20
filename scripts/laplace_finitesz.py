import numpy as np
import torch

import sys
import os

sys.path.append("../")

from FileManager import FileManager
from kernels import LaplaceKernel
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy, Hyperparams
from data import get_powerlaw, get_matrix_hermites


hypers = Hyperparams(
    expt_name = "laplace-finitesz",
    dataset = "gaussian",
    kernel_name = "LaplaceKernel",
    kernel_width = 4,
    n_samples = 20_000,
    p_modes = 20_000,
    # If using synth data, set these
    data_dim = 200,
    data_eigval_exp = 1.2,
    # If using natural image data, set these
    zca_strength = 0,
)

# SETUP FILE MANAGEMENT
#######################

datapath = os.getenv("DATASETPATH")
exptpath = os.getenv("EXPTPATH")
if datapath is None:
    raise ValueError("must set $DATASETPATH environment variable")
if exptpath is None:
    raise ValueError("must set $EXPTPATH environment variable")
expt_dir = os.path.join(exptpath, "phlab", hypers.expt_name, hypers.dataset)
expt_dir = os.path.join(expt_dir, hypers.generate_filepath())

if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)
expt_fm = FileManager(expt_dir)
print(f"Working in directory {expt_dir}.")
hypers.save(expt_fm.get_filename("hypers.json"))

# START EXPERIMENT
##################

assert hypers.dataset == "gaussian"
assert hypers.kernel_name == "LaplaceKernel"

data_eigvals = get_powerlaw(hypers.data_dim, hypers.data_eigval_exp, offset=6)
N, d = hypers.n_samples, hypers.data_dim
# on average, we expect norm(x_i) ~ Tr(data_eigvals)
X = ensure_torch(torch.normal(0, 1, (N, d))) * torch.sqrt(data_eigvals)
d_eff = 1/(data_eigvals**2).sum()

eval_level_coeff = LaplaceKernel.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                 kernel_width=hypers.kernel_width)
hehe_eigvals, monomials = generate_fra_monomials(data_eigvals, hypers.p_modes, eval_level_coeff)

kernel_sizes = np.logspace(1, np.log10(hypers.n_samples), num=10).astype(int)
emp_eigvals_n = {}
Ms_n = {}
print(f"n:", end=" ", flush=True)
for n in kernel_sizes:
    print(f"{n}", end=" ", flush=True)
    X_n = X[:n]
    kernel = LaplaceKernel(X_n, kernel_width=hypers.kernel_width)
    emp_eigvals, emp_eigvecs = kernel.eigendecomp()
    emp_eigvals_n[n] = ensure_numpy(emp_eigvals)
    H_n = get_matrix_hermites(X_n, monomials[:n])
    M_n = (emp_eigvecs.T @ H_n)**2
    Ms_n[n] = ensure_numpy(M_n)
    torch.cuda.empty_cache()
print()
    
expt_fm.save(emp_eigvals_n, "emp_eigvals_n.pickle")
expt_fm.save(Ms_n, "Ms_n.pickle")

result = {
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "th_eigvals": hehe_eigvals
}
expt_fm.save(result, "result.pickle")
