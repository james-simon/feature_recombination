import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange, reduce, repeat

import sys
import os

sys.path.append("../")

from ImageData import ImageData, preprocess
from ExptTrace import ExptTrace
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel, ExponentialKernel
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy
from data import get_powerlaw, get_gaussian_data, get_matrix_hermites
to_torch = ensure_torch


# TOP LEVEL HYPERPARAMS
#######################

EXPT_NAME = "blah"
DATASET = "cifar10"

N_SAMPLES = 20_000
P_MODES = 10_000
DATA_DIM = 200
KERNEL_WIDTH = 4

# SETUP FILE MANAGEMENT
#######################

datapath = os.getenv("DATASETPATH")
exptpath = os.getenv("EXPTPATH")
if datapath is None:
    raise ValueError("must set $DATASETPATH environment variable")
if exptpath is None:
    raise ValueError("must set $EXPTPATH environment variable")
expt_dir = os.path.join(exptpath, "phlab", EXPT_NAME, DATASET)

if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)
expt_fm = FileManager(expt_dir)

# START EXPERIMENT
##################

data_eigval_exps = np.linspace(1., 2., num=3)
zca_strengths = [0, 1e-3, 2e-2]
kerneltypes = [GaussianKernel, ExponentialKernel, LaplaceKernel]

var_axes = ["d_eff", "kernel"]
et_pathnames, et_emp_eigvals, et_fra_eigvals = ExptTrace.multi_init(3, var_axes)

if DATASET == "cifar10":
    cifar10 = ImageData('cifar10', datapath, classes=None)
    X_raw, _ = cifar10.get_dataset(N_SAMPLES, get="train")

for i in range(len(data_eigval_exps)):    
    if DATASET == "gaussian":
        data_eigval_exp = data_eigval_exps[i]
        data_eigvals = get_powerlaw(DATA_DIM, data_eigval_exp, offset=6)
        X = get_gaussian_data(N_SAMPLES, data_eigvals)
    if DATASET == "cifar10":
        zca_strength = zca_strengths[i]
        X = preprocess(X_raw, center=True, grayscale=True, zca_strength=zca_strength)
        X = ensure_torch(X)
        # ensure typical sample has unit norm
        S = torch.linalg.svdvals(X)
        X *= torch.sqrt(N_SAMPLES / (S**2).sum())
        data_eigvals = S**2 / (S**2).sum()
    
    d_eff = 1/(data_eigvals**2).sum()
    
    for kerneltype in kerneltypes:
        print(".", end="", flush=True)
        torch.cuda.empty_cache()
        kernelname = kerneltype.__name__
        
        subpath = f"{kernelname}-deff{d_eff:.2f}"
        expt_fm.set_filepath(subpath)
        et_pathnames[d_eff, kernelname] = subpath
        
        kernel = kerneltype(X, kernel_width=KERNEL_WIDTH)
        eigvals, eigvecs = kernel.eigendecomp()
        expt_fm.save(kernel.serialize(), "kernel.pickle")

        eval_level_coeff = kerneltype.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                        kernel_width=KERNEL_WIDTH)
        fra_eigvals, monomials = generate_fra_monomials(data_eigvals, P_MODES, eval_level_coeff)
        H = get_matrix_hermites(X, monomials)
        expt_fm.save([dict(m) for m in monomials], "monomials.pickle")
        expt_fm.save(ensure_numpy(H), "H.npy")
        
        et_emp_eigvals[d_eff, kernelname] = eigvals.cpu().numpy()
        et_fra_eigvals[d_eff, kernelname] = fra_eigvals

        # pdf, cdf, quartiles = kernel_hermite_overlap_estimation(kernel, H)
    print()

result = {
    "pathnames": et_pathnames.serialize(),
    "emp_eigvals": et_emp_eigvals.serialize(),
    "fra_eigvals": et_fra_eigvals.serialize(),
}
expt_fm.set_filepath("")
expt_fm.save(result, f"result.expt")