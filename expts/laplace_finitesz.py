import numpy as np
import torch

import sys
import os
import json

sys.path.append("../")

from FileManager import FileManager
from kernels import LaplaceKernel
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy, get_powerlaw
from data import get_matrix_hermites


## sample values
# kerneltypes = [GaussianKernel, LaplaceKernel]
# kernel_widths = [1, 4]
# data_eigval_exps = [1.2, 1.5, 2.]
# zca_strengths = [0, 5e-3, 3e-2]

EXPT_NAME = "laplace-finitesz"
DATASET = "gaussian"
KERNEL_TYPE = LaplaceKernel
KERNEL_WIDTH = 4
N_SAMPLES = 20_000
P_MODES = 20_000
DATA_DIM = 200
DATA_EIGVAL_EXP = 1.2
ZCA_STRENGTH = 1e-2

hypers = dict(expt_name=EXPT_NAME, dataset=DATASET, kernel_name=KERNEL_TYPE.__name__,
              kernel_width=KERNEL_WIDTH, n_samples=N_SAMPLES, p_modes=P_MODES,
              data_dim=DATA_DIM, data_eigval_exp=DATA_EIGVAL_EXP,
              zca_strength=ZCA_STRENGTH)

# SETUP FILE MANAGEMENT
#######################

datapath = os.getenv("DATASETPATH")
exptpath = os.getenv("EXPTPATH")
if datapath is None:
    raise ValueError("must set $DATASETPATH environment variable")
if exptpath is None:
    raise ValueError("must set $EXPTPATH environment variable")
fp = f"{KERNEL_TYPE.__name__}-kw:{KERNEL_WIDTH}"
if DATASET == "gaussian":
    fp += f"-data:{DATA_DIM}:{DATA_EIGVAL_EXP}"
else:
    fp += f"-zca:{ZCA_STRENGTH}"
expt_dir = os.path.join(exptpath, "phlab", EXPT_NAME, DATASET, fp)

if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)
expt_fm = FileManager(expt_dir)
print(f"Working in directory {expt_dir}.")
with open(expt_fm.get_filename("hypers.json"), 'w') as f:
    json.dump(hypers, f, indent=4)

# START EXPERIMENT
##################

assert DATASET == "gaussian"
assert KERNEL_TYPE.__name__ == "LaplaceKernel"

data_eigvals = get_powerlaw(DATA_DIM, DATA_EIGVAL_EXP, offset=6)
# on average, we expect norm(x_i) ~ Tr(data_eigvals)
X = ensure_torch(torch.normal(0, 1, (N_SAMPLES, DATA_DIM)))
X *= torch.sqrt(data_eigvals)

d_eff = 1/(data_eigvals**2).sum().item()
print(f"d_eff: {d_eff:.2f}", end="\n")

eval_level_coeff = LaplaceKernel.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                 kernel_width=KERNEL_WIDTH)
hehe_eigvals, monomials = generate_fra_monomials(data_eigvals, P_MODES, eval_level_coeff)

kernel_sizes = np.logspace(1, np.log10(N_SAMPLES), num=10).astype(int)
emp_eigvals_n = {}
Ms_n = {}
print(f"n:", end=" ", flush=True)
for n in kernel_sizes:
    print(f"{n}", end=" ", flush=True)
    X_n = X[:n]
    kernel = LaplaceKernel(X_n, kernel_width=KERNEL_WIDTH)
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
