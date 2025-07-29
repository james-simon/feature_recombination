import numpy as np
import torch

from einops import rearrange

import sys
import os

sys.path.append("../")

from ImageData import ImageData, preprocess
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy, Hyperparams
from data import get_powerlaw, get_matrix_hermites


## sample values
# kerneltypes = [GaussianKernel, LaplaceKernel]
# kernel_widths = [1, 4]
# data_eigval_exps = [1., 1.5, 2.]
# zca_strengths = [0, 5e-3, 3e-2]

hypers = Hyperparams(
    expt_name = "verify-hehe",
    dataset = "imagenet32",
    kernel_name = "GaussianKernel",
    kernel_width = 4,
    n_samples = 20_000,
    p_modes = 10_000,
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

if hypers.dataset == "gaussian":
    data_eigvals = get_powerlaw(hypers.data_dim, hypers.data_eigval_exp, offset=6)
    N, d = hypers.n_samples, hypers.data_dim
    # on average, we expect norm(x_i) ~ Tr(data_eigvals)
    X = ensure_torch(torch.normal(0, 1, (N, d))) * torch.sqrt(data_eigvals)
if hypers.dataset in ["cifar10", "imagenet32"]:
    if hypers.dataset == "cifar10":
        data_dir = os.path.join(datapath, "cifar10")
        cifar10 = ImageData('cifar10', data_dir, classes=None)
        X_raw, _ = cifar10.get_dataset(hypers.n_samples, get="train")
    if hypers.dataset == "imagenet32":
        fn = os.path.join(datapath, "imagenet", f"{hypers.dataset}.npz")
        data = np.load(fn)
        X_raw = data['data'][:hypers.n_samples].astype(float)
        X_raw = rearrange(X_raw, 'n (c h w) -> n c h w', c=3, h=32, w=32)
    X = preprocess(X_raw, center=True, grayscale=True, zca_strength=hypers.zca_strength)
    X = ensure_torch(X)
    # ensure typical sample has unit norm
    S = torch.linalg.svdvals(X)
    X *= torch.sqrt(hypers.n_samples / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()

d_eff = 1/(data_eigvals**2).sum()

kerneltype = {
    "GaussianKernel": GaussianKernel,
    "LaplaceKernel": LaplaceKernel
}[hypers.kernel_name]
kernel = kerneltype(X, kernel_width=hypers.kernel_width)
emp_eigvals, _ = kernel.eigendecomp()
expt_fm.save(kernel.serialize(), "kernel.pickle")

eval_level_coeff = kerneltype.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                 kernel_width=hypers.kernel_width)
hehe_eigvals, monomials = generate_fra_monomials(data_eigvals, hypers.p_modes, eval_level_coeff)

H = get_matrix_hermites(X, monomials)
expt_fm.save(ensure_numpy(H), "H.npy")

result = {
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "emp_eigvals": ensure_numpy(emp_eigvals),
    "th_eigvals": hehe_eigvals
}
expt_fm.save(result, "result.pickle")
