import numpy as np
import torch

from einops import rearrange

import sys
import os
import json

sys.path.append("../")

from ImageData import ImageData, preprocess
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy
from data import get_powerlaw, get_matrix_hermites


## sample values
# kerneltypes = [GaussianKernel, LaplaceKernel]
# kernel_widths = [1, 4]
# data_eigval_exps = [1.2, 1.5, 2.]
# zca_strengths = [0, 5e-3, 3e-2]

EXPT_NAME = "verify-hehe"
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

if DATASET == "gaussian":
    data_eigvals = get_powerlaw(DATA_DIM, DATA_EIGVAL_EXP, offset=6)
    # on average, we expect norm(x_i) ~ Tr(data_eigvals)
    X = ensure_torch(torch.normal(0, 1, (N_SAMPLES, DATA_DIM)))
    X *= torch.sqrt(data_eigvals)
if DATASET in ["cifar10", "imagenet32"]:
    if DATASET == "cifar10":
        data_dir = os.path.join(datapath, "cifar10")
        cifar10 = ImageData('cifar10', data_dir, classes=None)
        X_raw, _ = cifar10.get_dataset(N_SAMPLES, get="train")
    if DATASET == "imagenet32":
        fn = os.path.join(datapath, "imagenet", f"{DATASET}.npz")
        data = np.load(fn)
        X_raw = data['data'][:N_SAMPLES].astype(float)
        X_raw = rearrange(X_raw, 'n (c h w) -> n c h w', c=3, h=32, w=32)
    X = preprocess(X_raw, center=True, grayscale=True, zca_strength=ZCA_STRENGTH)
    X = ensure_torch(X)
    # ensure typical sample has unit norm
    S = torch.linalg.svdvals(X)
    X *= torch.sqrt(N_SAMPLES / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()

d_eff = 1/(data_eigvals**2).sum().item()
print(f"d_eff: {d_eff:.2f}", end="\n")

kernel = KERNEL_TYPE(X, kernel_width=KERNEL_WIDTH)
emp_eigvals, _ = kernel.eigendecomp()
expt_fm.save(kernel.serialize(), "kernel.pickle")

eval_level_coeff = KERNEL_TYPE.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                  kernel_width=KERNEL_WIDTH)
hehe_eigvals, monomials = generate_fra_monomials(data_eigvals, P_MODES, eval_level_coeff)
H = get_matrix_hermites(X, monomials)
expt_fm.save(ensure_numpy(H), "H.npy")

result = {
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "emp_eigvals": ensure_numpy(emp_eigvals),
    "th_eigvals": hehe_eigvals
}
expt_fm.save(result, "result.pickle")
