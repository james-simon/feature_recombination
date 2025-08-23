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
from feature_decomp import generate_fra_monomials
from utils import ensure_torch, ensure_numpy
from data import get_powerlaw, get_matrix_hermites, get_powerlaw_target


EXPT_NAME = "hehe-eigenlearning"
KERNEL_WIDTH = 4
N_SAMPLES = 25_000
P_MODES = 25_000
DATA_DIM = 200
TARGET = "powerlaws"

# Allow command line argument
if len(sys.argv) == 1:
    DATASET = "gaussian"
    KERNEL_TYPE = GaussianKernel
else:    
    try:
        expt_id = int(sys.argv[1])
        if expt_id == 1:
            DATASET = "gaussian"
            KERNEL_TYPE = GaussianKernel
        elif expt_id == 2:
            DATASET = "gaussian"
            KERNEL_TYPE = LaplaceKernel
        elif expt_id == 3:
            DATASET = "cifar10"
            KERNEL_TYPE = GaussianKernel
        elif expt_id == 4:
            DATASET = "cifar10"
            KERNEL_TYPE = LaplaceKernel
        else:
            raise ValueError("Invalid expt number")
    except ValueError:
        print("Error: Argument must be an integer")
        sys.exit(1)

if KERNEL_TYPE == GaussianKernel:
    DATA_EIGVAL_EXP = 2.0   # d_eff = 15
    ZCA_STRENGTH = 5e-3     # d_eff = 18
if KERNEL_TYPE == LaplaceKernel:
    DATA_EIGVAL_EXP = 1.6   # d_eff = 27
    ZCA_STRENGTH = 1e-2     # d_eff = 26

hypers = dict(expt_name=EXPT_NAME, dataset=DATASET, kernel_name=KERNEL_TYPE.__name__,
              kernel_width=KERNEL_WIDTH, n_samples=N_SAMPLES, p_modes=P_MODES,
              data_dim=DATA_DIM, data_eigval_exp=DATA_EIGVAL_EXP,
              zca_strength=ZCA_STRENGTH,
              target=TARGET)

source_exps = [1.01, 1.1, 1.25, 1.5, 2.0]
target_monomials = [{10:1}, {190:1}, {0:2}, {1:1, 2:1}, {20:1,30:1}, {0:3}, {1:2, 2:1}]

# SETUP FILE MANAGEMENT
#######################

datapath = os.getenv("DATASETPATH")
exptpath = os.getenv("EXPTPATH")
if datapath is None:
    raise ValueError("must set $DATASETPATH environment variable")
if exptpath is None:
    raise ValueError("must set $EXPTPATH environment variable")
fp = f"{KERNEL_TYPE.__name__}-kw:{KERNEL_WIDTH}-target:{TARGET}"
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
    X *= torch.sqrt(ensure_torch(data_eigvals))
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
K = ensure_torch(kernel.K)

eval_level_coeff = KERNEL_TYPE.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                  kernel_width=KERNEL_WIDTH)
hehe_eigvals, monomials = generate_fra_monomials(data_eigvals, P_MODES, eval_level_coeff)
H = get_matrix_hermites(X, monomials[:P_MODES])

targets = {}
if TARGET == "powerlaws":
    for source_exp in source_exps:
        # get_powerlaw_target ensures size(y_i) ~ 1
        ystar = get_powerlaw_target(H, source_exp)
        targets[source_exp] = ensure_numpy(ystar)
if TARGET == "monomials":
    monomial_idxs = []
    for tmon in target_monomials:
        try:
            monomial_idxs.append(monomials.index(tmon))
        except ValueError:
            print(f"Warning: target {tmon} not in generated monomials. Skipping.")
    assert len(monomial_idxs) > 0
    
    if sorted(monomial_idxs) == monomial_idxs:
        print("target monomials are monotonic :(")
    for idx in monomial_idxs:
        ystar = ensure_numpy(H[:, idx])
        # ensure size(y_i) ~ 1
        targets[idx] = np.sqrt(N_SAMPLES) * ystar / np.linalg.norm(ystar)
    print(f"target idxs: {monomial_idxs}")
    # print(f"target eigvals: {hehe_eigvals[monomial_idxs]}")
if TARGET == "true":
    pass

def get_ntrials(ntrain):
    if ntrain < 100: return 20
    elif ntrain < 1000: return 10
    elif ntrain < 10000: return 5
    else: return 2

ntest = 5000
log_ntrain_max = np.log10((N_SAMPLES - ntest)/1.1)
ntrains = np.logspace(1, log_ntrain_max, base=10, num=30).astype(int)
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

torch.cuda.empty_cache()
emp_eigvals, emp_eigvecs = kernel.eigendecomp()
expt_fm.save(ensure_numpy(emp_eigvecs), "emp_eigvecs.npy")
expt_fm.save(ensure_numpy(H), "H.npy")
expt_fm.save(targets, "targets.pickle")
iso_data_eigvals = torch.ones_like(data_eigvals) / len(data_eigvals)
iso_eigvals, _ = generate_fra_monomials(iso_data_eigvals, P_MODES, eval_level_coeff)
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
torch.cuda.empty_cache()