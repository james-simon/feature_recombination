import numpy as np
import torch

from einops import rearrange

import sys
import os
import json

sys.path.append("../")

from ImageData import ImageData, preprocess
from ExptTrace import ExptTrace
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel, krr
from feature_decomp import generate_hea_monomials
from utils import ensure_torch, ensure_numpy
from data import get_powerlaw, get_matrix_hermites


EXPT_NAME = "compare-monomials"
KERNEL_WIDTH = 10
N_SAMPLES = 26_000
P_MODES = 100_000
DATA_DIM = 200
NUM__MARKERS = 20

# Defaults
DATASET = "gaussian"
KERNEL_TYPE = GaussianKernel

# Allow command line arguments
if len(sys.argv) > 1:
    try:
        expt_id = int(sys.argv[1])
    except ValueError:
        print("Error: Expt num must be an integer")
        sys.exit(1)

    if expt_id in [1, 2]:   DATASET = "gaussian"
    elif expt_id in [3, 4]: DATASET = "cifar10"
    elif expt_id in [5, 6]: DATASET = "svhn"
    else:                   raise ValueError("Invalid expt number")

    if expt_id % 2 == 1:    KERNEL_TYPE = GaussianKernel
    else:                   KERNEL_TYPE = LaplaceKernel

if KERNEL_TYPE == GaussianKernel:
    RIDGE = 1e-3
    DATA_EIGVAL_EXP = 3.0   # d_eff = 7
    ZCA_STRENGTH = 0        # d_eff = 7 cf10
    if DATASET == "svhn":
        ZCA_STRENGTH = 5e-3 # d_eff = 9
if KERNEL_TYPE == LaplaceKernel:
    RIDGE = 1e-3
    DATA_EIGVAL_EXP = 1.6   # d_eff = 27
    ZCA_STRENGTH = 1e-2     # d_eff = 26 cf10

hypers = dict(expt_name=EXPT_NAME, dataset=DATASET, kernel_name=KERNEL_TYPE.__name__,
              kernel_width=KERNEL_WIDTH, n_samples=N_SAMPLES, p_modes=P_MODES,
              data_dim=DATA_DIM, data_eigval_exp=DATA_EIGVAL_EXP,
              zca_strength=ZCA_STRENGTH,
              ridge=RIDGE)


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


# SETUP EXPERIMENT
##################

if DATASET == "gaussian":
    data_eigvals = get_powerlaw(DATA_DIM, DATA_EIGVAL_EXP, offset=6)
    # on average, we expect norm(x_i) ~ Tr(data_eigvals)
    X = ensure_torch(torch.normal(0, 1, (N_SAMPLES, DATA_DIM)))
    X *= torch.sqrt(ensure_torch(data_eigvals))
if DATASET in ["cifar10", "imagenet32", "svhn"]:
    if DATASET == "cifar10":
        data_dir = os.path.join(datapath, "cifar10")
        cifar10 = ImageData('cifar10', data_dir, classes=None)
        X_raw, labels = cifar10.get_dataset(N_SAMPLES, get="train")
    if DATASET == "svhn":
        data_dir = os.path.join(datapath, "svhn")
        svhn = ImageData('svhn', data_dir, classes=None)
        X_raw, labels = svhn.get_dataset(N_SAMPLES, get="train")
    if DATASET == "imagenet32":
        fn = os.path.join(datapath, "imagenet", f"{DATASET}.npz")
        data = np.load(fn)
        X_raw = data['data'][:N_SAMPLES].astype(float)
        X_raw = rearrange(X_raw, 'n (c h w) -> n c h w', c=3, h=32, w=32)
    X = preprocess(X_raw, center=True, grayscale=False, zca_strength=ZCA_STRENGTH)
    X = ensure_torch(X)
    # ensure typical sample has unit norm
    S = torch.linalg.svdvals(X)
    X *= torch.sqrt(N_SAMPLES / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()

d_eff = 1/(data_eigvals**2).sum().item()
print(f"d_eff: {d_eff:.2f}", end="\n")

kernel = KERNEL_TYPE(X, kernel_width=KERNEL_WIDTH)
K = ensure_torch(kernel.K)

eval_level_coeff = KERNEL_TYPE.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                  kernel_width=KERNEL_WIDTH)
print("Generating hermites...", end=" ")
hehe_eigvals, monomials = generate_hea_monomials(data_eigvals, P_MODES, eval_level_coeff)
H = get_matrix_hermites(X, monomials)
print("done.")

target_monomials = [{0:1}, {100:1}, {0:2}, {0:1,1:1}, {16:1,20:1},
                    {0:3}, {0:1, 1:1, 2:1}, {3:2, 5:1},
                    {0:4}]
monomial_idxs = set()
for tmon in target_monomials:
    try:
        monomial_idxs.add(monomials.index(tmon))
    except ValueError:
        print(f"Warning: target {tmon} not in generated monomials. Skipping.")
assert len(monomial_idxs) > 0

# Add more modes: log-equidistant selection from hehe_eigvals
num_degrees = 4
masked_eigvals = np.ma.masked_all((num_degrees, len(hehe_eigvals)))
for idx, monomial in enumerate(monomials):
    if 1 <= monomial.degree() <= 4:
        masked_eigvals[monomial.degree()-1, idx] = hehe_eigvals[idx]
markers = np.logspace(np.log10(hehe_eigvals[-10_000]), np.log10(hehe_eigvals[0]), NUM__MARKERS)
for i, marker in enumerate(markers):
    degree = i%4 + 1
    idx = int(np.argmin(np.abs(masked_eigvals[degree-1] - marker)))
    if idx not in monomial_idxs:
        monomial_idxs.add(idx)
monomial_idxs = sorted(monomial_idxs)
print(f"Total target monomials: {len(monomial_idxs)}")

targets = {}
for idx in monomial_idxs:
    ystar = ensure_numpy(H[:, idx])
    # ensure size(y_i) ~ 1
    targets[idx] = np.sqrt(N_SAMPLES) * ystar / np.linalg.norm(ystar)
print(f"targets: {[str(monomials[idx]) for idx in monomial_idxs]}")

del H
torch.cuda.empty_cache()


# START EXPERIMENT
##################

print()

def get_ntrials(ntrain):
    if ntrain < 100: return 20
    elif ntrain < 1000: return 10
    elif ntrain < 10000: return 5
    else: return 2

ntest = 5_000
log_ntrain_max = np.log10((N_SAMPLES - ntest)/1.1)
ntrains = np.logspace(1, log_ntrain_max, base=10, num=30).astype(int)

et_yhat = ExptTrace(["trial", "ntrain", "target"])
for idx, ystar in targets.items():
    print("Starting target: ", str(monomials[idx]))
    ystar = ensure_torch(ystar)
    print(f"ridge={RIDGE}, ntrains:", end=" ", flush=True)
    for ntrain in ntrains:
        print(f"{ntrain}", end=" ", flush=True)
        for trial in range(get_ntrials(ntrain)):
            (y_hat, _), _ = krr(K, ystar, ntrain, n_test=ntest, ridge=RIDGE)
            et_yhat[trial, ntrain, idx] = y_hat.cpu().numpy()
    print()

print("diagonalizing kernel...", end=" ")
emp_eigvals = kernel.eigenvals()
print("saving results...", end=" ")
expt_fm.save(targets, "targets.pickle")
result = {
    "ridge": RIDGE,
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "n_test": ntest,
    "emp_eigvals": ensure_numpy(emp_eigvals),
    "th_eigvals": hehe_eigvals,
    "y_hat": et_yhat.serialize()
}
expt_fm.save(result, "result.pickle")
torch.cuda.empty_cache()
print("done.")
