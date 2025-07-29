import numpy as np
import torch

from einops import rearrange

import sys
import os

sys.path.append("../")

from ImageData import ImageData, preprocess
from ExptTrace import ExptTrace
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel, krr
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy, Hyperparams
from data import get_powerlaw, get_matrix_hermites, get_powerlaw_target

hypers = Hyperparams(
    expt_name = "hehe-eigenlearning",
    dataset = "gaussian",
    kernel_name = "GaussianKernel",
    kernel_width = 4,
    n_samples = 20_000,
    p_modes = 20_000,
    # If using synth data, set these
    data_dim = 100,
    data_eigval_exp = 1.4,
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
expt_dir = os.path.join(expt_dir, hypers.generate_filepath())

if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)
expt_fm = FileManager(expt_dir)
print(f"Working in directory {expt_dir}.")
hypers.save(expt_fm.get_filename("hypers.json"))

# START EXPERIMENT
##################

kerneltype = {
    "GaussianKernel": GaussianKernel,
    "LaplaceKernel": LaplaceKernel
}[hypers.kernel_name]

if hypers.dataset == "gaussian":
    data_eigvals = get_powerlaw(hypers.data_dim, hypers.data_eigval_exp, offset=6)
    N, d = hypers.n_samples, hypers.data_dim
    # on average, we expect norm(x_i) ~ Tr(data_eigvals)
    X = ensure_torch(torch.normal(0, 1, (N, d))) * torch.sqrt(ensure_torch(data_eigvals))

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

eval_level_coeff = kerneltype.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                 kernel_width=hypers.kernel_width)
hehe_eigvals, monomials = generate_fra_monomials(data_eigvals, hypers.p_modes, eval_level_coeff)
H = get_matrix_hermites(X, monomials[:hypers.p_modes])

if True or hypers.dataset == "gaussian":
    targets = {}
    source_exps = [1.01, 1.25, 1.5, 2.0]
    for source_exp in source_exps:
        ystar = get_powerlaw_target(H, source_exp)
        targets[source_exp] = ensure_numpy(ystar)
if hypers.dataset == "cifar10":
    pass
if hypers.dataset == "imagenet32":
    pass

kernel = kerneltype(X, kernel_width=hypers.kernel_width)
K = ensure_torch(kernel.K)

def get_ntrials(ntrain):
    if ntrain < 100: return 20
    elif ntrain < 1000: return 10
    elif ntrain < 10000: return 5
    else: return 2

ntest = 5000
log_ntrain_max = np.log10((hypers.n_samples - ntest) / 2)
ntrains = np.logspace(1, log_ntrain_max, base=10, num=50).astype(int)
ridges = [1e-4]

var_axes = ["trial", "ntrain", "ridge", "target"]
et_yhat = ExptTrace(var_axes)

for target, ystar in targets.items():
    print("Starting target: ", target)
    ystar = ensure_torch(ystar)
    for ridge in ridges:
        print(f"ridge={ridge}, ntrains:", end=" ", flush=True)
        for ntrain in ntrains:
            print(f"{ntrain}", end=" ", flush=True)
            for trial in range(get_ntrials(ntrain)):
                (y_hat, y_test), _ = krr(K, ystar, ntrain, n_test=ntest, ridge=ridge)
                et_yhat[trial, ntrain, ridge, target] = y_hat.cpu().numpy()
        print()
    print()

emp_eigvals, emp_eigvecs = kernel.eigendecomp()
expt_fm.save(ensure_numpy(emp_eigvecs), "emp_eigvecs.npy")
expt_fm.save(ensure_numpy(H), "H.npy")
expt_fm.save(targets, "targets.pickle")
iso_data_eigvals = torch.ones_like(data_eigvals) / len(data_eigvals)
iso_eigvals, _ = generate_fra_monomials(iso_data_eigvals, hypers.p_modes, eval_level_coeff)
result = {
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "n_test": ntest,
    "emp_eigvals": ensure_numpy(emp_eigvals),
    "th_eigvals": hehe_eigvals,
    "iso_eigvals": ensure_numpy(iso_eigvals),
    "y_hat": et_yhat.serialize()
}
expt_fm.save(result, "result.pickle")
