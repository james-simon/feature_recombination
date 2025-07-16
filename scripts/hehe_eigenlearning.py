import numpy as np
import torch

from einops import rearrange, reduce, repeat

import sys
import os

sys.path.append("../")

from ImageData import ImageData, preprocess
from ExptTrace import ExptTrace
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel, ExponentialKernel
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy, Hyperparams
from data import get_powerlaw, get_gaussian_data, get_matrix_hermites, get_hermite_target
to_torch = ensure_torch

hypers = Hyperparams(
    expt_name = "hehe-eigenlearning-tests",
    dataset = "cifar10",
    kernel_name = "GaussianKernel",
    kernel_width = 4,
    n_samples = 20_000,
    p_modes = 10_000,
    # If using synth data, set these
    data_dim = 200,
    data_eigval_exp = 1.2,
    beta = 1.1,
    noise_var = 0.1,
    # If using natural image data, set these
    zca_strength = 5e-3,
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

if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)
expt_fm = FileManager(expt_dir)

# START EXPERIMENT
##################

var_axes = ["trial", "ntrain", "ridge"]
et_pathnames, et_emp_eigvals, et_fra_eigvals = ExptTrace.multi_init(3, var_axes)

if hypers.dataset == "gaussian":
    data_eigvals = get_powerlaw(hypers.data_dim, hypers.data_eigval_exp, offset=6)
    X = get_gaussian_data(hypers.n_samples, data_eigvals)

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

subpath = f"{hypers.kernel_name}-deff{d_eff:.2f}"
expt_fm.set_filepath(subpath)
et_pathnames[d_eff, kernelname, kernel_width] = subpath

kernel = kerneltype(X, kernel_width=kernel_width)
eigvals, eigvecs = kernel.eigendecomp()
expt_fm.save(kernel.serialize(), "kernel.pickle")

eval_level_coeff = kerneltype.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                    kernel_width=kernel_width)
fra_eigvals, monomials = generate_fra_monomials(data_eigvals, P_MODES, eval_level_coeff)
H = get_matrix_hermites(X, monomials)
expt_fm.save([dict(m) for m in monomials], "monomials.pickle")
expt_fm.save(ensure_numpy(H), "H.npy")

squared_coeffs = get_powerlaw(P_MODES, BETA, offset=6)
y, _ = get_hermite_target(H, squared_coeffs, noise_var=NOISE_VAR)

ntrains = np.logspace(1, 4, base=10, num=20).astype(int)
et_test_mse = ExptTrace(["trial", "n"])
et_train_mse = ExptTrace(["trial", "n"])
ystar_idx = 5
ridge = 1e-3
ntrials = 5

K = ensure_torch(kernel.K)

for trial in range(ntrials):
    for ntrain in ntrains:
        train_mse, test_mse, yhattest = krr(K, y, ntrain, n_test=2000, ridge=ridge)
        et_test_mse[trial, ntrain] = test_mse
        et_train_mse[trial, ntrain] = train_mse
et_emp_eigvals[d_eff, kernelname, kernel_width] = eigvals.cpu().numpy()
et_fra_eigvals[d_eff, kernelname, kernel_width] = fra_eigvals


result = {
    "pathnames": et_pathnames.serialize(),
    "emp_eigvals": et_emp_eigvals.serialize(),
    "fra_eigvals": et_fra_eigvals.serialize(),
}
expt_fm.set_filepath("")
expt_fm.save(result, f"result.expt")