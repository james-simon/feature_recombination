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
from data import get_powerlaw, get_matrix_hermites, get_powerlaw_target
from tools import grf


EXPT_NAME = "learning-curves-debug"
N_SAMPLES = 25_000
P_MODES = 50_000
DATA_DIM = 200

# Defaults
DATASET = "gaussian"
DATA_EIGVAL_EXP = 1.0
ZCA_STRENGTH = 0
GRAYSCALE = False
M_GRF = 99_000
TARGET = "powerlaws"
source_exps = [1.15]
KERNEL_TYPE = GaussianKernel
KERNEL_WIDTH = 10
RIDGE = 1e-3

# Allow command line arguments
if len(sys.argv) > 1:
    try:
        expt_id = int(sys.argv[1])
    except ValueError:
        print("Error: Expt num must be an integer")
        sys.exit(1)
    
    if expt_id == 1:
        DATASET = "cifar10"
        TARGET = "vehicle"
    if expt_id == 2:
        DATASET = "cifar10"
        TARGET = "domesticated"
    if expt_id == 3:
        DATASET = "svhn"
        ZCA_STRENGTH = 1e-2
        TARGET = "evenodd"
        GRAYSCALE = True
    if expt_id == 4:
        DATASET = "svhn"
        ZCA_STRENGTH = 1e-2
        TARGET = "loops"
        GRAYSCALE = True
    if expt_id == 5:
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 5e-3
        DATASET = "cifar10"
        TARGET = "vehicle"
        M_GRF = 40_000
    if expt_id == 6:
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 5e-3
        DATASET = "cifar10"
        TARGET = "domesticated"
        M_GRF = 30_000
    if expt_id == 7:
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 1e-2
        GRAYSCALE = True
        DATASET = "svhn"
        TARGET = "evenodd"
    if expt_id == 8:
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 1e-2
        GRAYSCALE = True
        DATASET = "svhn"
        TARGET = "loops"
    if expt_id == 9:
        # THIS WORKS!
        KERNEL_TYPE = LaplaceKernel
        DATASET = "gaussian"
        DATA_EIGVAL_EXP = 1.6
        TARGET = "powerlaws"
    if expt_id == 10:
        # THIS WORKS!
        KERNEL_TYPE = LaplaceKernel
        ZCA_STRENGTH = 1e-2
        DATASET = "cifar10"
        M_GRF = 50_000
        TARGET = "powerlaws"
    
assert TARGET in ["powerlaws", "original", "vehicle", "domesticated", "evenodd", "loops"]

hypers = dict(expt_name=EXPT_NAME, dataset=DATASET, kernel_name=KERNEL_TYPE.__name__,
              kernel_width=KERNEL_WIDTH, n_samples=N_SAMPLES, p_modes=P_MODES,
              data_dim=DATA_DIM, data_eigval_exp=DATA_EIGVAL_EXP,
              zca_strength=ZCA_STRENGTH, grayscale=GRAYSCALE,
              ridge=RIDGE, target=TARGET)


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


# SETUP EXPERIMENT
##################

if DATASET == "gaussian":
    data_eigvals = get_powerlaw(DATA_DIM, DATA_EIGVAL_EXP, offset=6)
    # on average, we expect norm(x_i) ~ Tr(data_eigvals)
    X = ensure_torch(torch.normal(0, 1, (M_GRF, DATA_DIM)))
    X *= torch.sqrt(ensure_torch(data_eigvals))
if DATASET in ["cifar10", "imagenet32", "svhn"]:
    if DATASET == "cifar10":
        data_dir = os.path.join(datapath, "cifar10")
        if TARGET == "vehicle":
            # plane car ship truck vs bird cat deer dog
            classes = [[0, 1, 8, 9], [2, 3, 4, 5]]
        elif TARGET == "domesticated":
            # cat dog horse vs bird deer frog
            classes = [[3, 5, 7], [2, 4, 6]]
        else: classes = None
        cifar10 = ImageData('cifar10', data_dir, classes=classes)
        X_raw, labels = cifar10.get_dataset(M_GRF, get="train")
    if DATASET == "svhn":
        data_dir = os.path.join(datapath, "svhn")
        if TARGET == "evenodd":
            classes = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
        elif TARGET == "loops":
            classes = [[0, 6, 8, 9], [1, 3, 5, 7]]
        else: classes = None
        svhn = ImageData('svhn', data_dir, classes=classes)
        if M_GRF > 73_257:
            assert M_GRF <= 73_257 + 26_032, "SVHN dataset size exceeded"
            # TODO
        else:
            X_raw, labels = svhn.get_dataset(M_GRF, get="train")
    if DATASET == "imagenet32":
        fn = os.path.join(datapath, "imagenet", f"{DATASET}.npz")
        data = np.load(fn)
        X_raw = data['data'][:M_GRF].astype(float)
        X_raw = rearrange(X_raw, 'n (c h w) -> n c h w', c=3, h=32, w=32)
    X = preprocess(X_raw, center=True, grayscale=GRAYSCALE, zca_strength=ZCA_STRENGTH)
    X = ensure_torch(X)
    # ensure typical sample has unit norm
    S = torch.linalg.svdvals(X)
    X *= torch.sqrt(M_GRF / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()

d_eff = 1/(data_eigvals**2).sum()
print(f"d_eff: {d_eff:.2f}", end="\n")

eval_level_coeff = KERNEL_TYPE.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                  kernel_width=KERNEL_WIDTH)
print("Generating hermites...", end=" ")
hehe_eigvals, monomials = generate_hea_monomials(data_eigvals, P_MODES, eval_level_coeff)
H = get_matrix_hermites(X, monomials)
print("done.")

targets = {}
if TARGET == "powerlaws":
    for source_exp in source_exps:
        # get_powerlaw_target ensures size(y_i) ~ 1
        ystar = get_powerlaw_target(H, source_exp)
        targets[source_exp] = ystar
if TARGET == "original":
    for i in range(10):
        # z-score normalization ensures size(y_i) ~ 1
        targets[i] = ensure_numpy(1/3 * (-1 + 10*labels[:, i]))
if TARGET in ["vehicle", "domesticated", "evenodd", "loops"]:
    targets[TARGET] = ensure_numpy(-1 + 2*labels[:, 1])

X_krn = X[:N_SAMPLES, :]
kernel = KERNEL_TYPE(X_krn, kernel_width=KERNEL_WIDTH)
K = ensure_torch(kernel.K)


# START GRF
##################

print()

grf_results = {
    "coeffs": ExptTrace(["target"]),
    "residual": ExptTrace(["target"]),
}
for target, ystar in targets.items():
    ystar = ystar / np.linalg.norm(ystar)
    print(f"Solving GRF coeffs for target {target}...", end=" ")
    # coeffs, _, _ = grf(H, ystar)
    # sorted_idx = np.argsort(-coeffs**2)
    # coeffs, residual, H_norm = grf(H, ystar, idxs=sorted_idx)
    coeffs, residual, H_norm = grf(H, ystar)
    grf_results["coeffs"][target] = coeffs
    grf_results["residual"][target] = residual
    print("done.")

grf_results = {k: v.serialize() for k, v in grf_results.items()}
expt_fm.save(grf_results, "grf.pickle")
# expt_fm.save(H, "H.npy")
expt_fm.save(H_norm, "H_norm.npy")
del H


# START EXPERIMENT
##################

print()

def get_ntrials(ntrain):
    if ntrain < 100: return 50
    elif ntrain < 2000: return 10
    return max(1, N_SAMPLES // ntrain)

ntest = 5_000 # DEBUG 10_000
log_ntrain_max = np.log10((N_SAMPLES - ntest)/1.1)
ntrains = np.logspace(1, log_ntrain_max, base=10, num=30).astype(int)

et_yhat = ExptTrace(["trial", "ntrain", "target"])
for target, ystar in targets.items():
    print("Starting target: ", target)
    ystar = ensure_torch(ystar[:N_SAMPLES])
    print(f"ridge={RIDGE}, ntrains:", end=" ", flush=True)
    for ntrain in ntrains:
        print(f"{ntrain}", end=" ", flush=True)
        for trial in range(get_ntrials(ntrain)):
            (y_hat, _), _ = krr(K, ystar, ntrain, n_test=ntest, ridge=RIDGE, trial=trial)
            et_yhat[trial, ntrain, target] = y_hat.cpu().numpy()
    print()

# print("diagonalizing kernel...", end=" ")
# emp_eigvals, emp_eigvecs = kernel.eigendecomp()
print("saving results...", end=" ")
# expt_fm.save(ensure_numpy(emp_eigvecs), "emp_eigvecs.npy")
expt_fm.save(targets, "targets.pickle")
# iso_data_eigvals = torch.ones(len(data_eigvals)) / len(data_eigvals)
# iso_eigvals, _ = generate_hea_monomials(iso_data_eigvals, P_MODES, eval_level_coeff)
result = {
    "ridge": RIDGE,
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "n_test": ntest,
    # "emp_eigvals": ensure_numpy(emp_eigvals),
    "th_eigvals": hehe_eigvals,
    # "iso_eigvals": ensure_numpy(iso_eigvals),
    "y_hat": et_yhat.serialize()
}
expt_fm.save(result, "result.pickle")
torch.cuda.empty_cache()
print("done.")
