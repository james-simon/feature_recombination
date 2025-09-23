import numpy as np
import torch

import sys
import os
import json

sys.path.append("../")

from ExptTrace import ExptTrace
from FileManager import FileManager
from kernels import GaussianKernel, LaplaceKernel, ReluNTK, krr_loop
from feature_decomp import generate_hea_monomials, get_monomial_targets
from utils import ensure_torch, ensure_numpy, get_powerlaw
from data import get_binarized_dataset, preprocess, compute_hermite_basis
from expt_demux import expt_demux


EXPT_NAME = "learning-curves"
expt_id = None
if len(sys.argv) > 1:
    try:
        expt_id = int(sys.argv[1])
    except ValueError:
        print("Error: Expt num must be an integer")
        sys.exit(1)
hypers = expt_demux(expt_id)

N_SAMPLES = hypers['n_samples']
N_KERNEL = hypers['n_kernel']
N_TRAIN_MAX = hypers['n_train_max']
N_TEST = hypers['n_test']
P_MODES = hypers['p_modes']
DATASET = hypers['dataset']
DATA_DIM = hypers['data_dim']
DATA_EIGVAL_EXP = hypers['data_eigval_exp']
ZCA_STRENGTH = hypers['zca_strength']
NORMALIZE = hypers['normalize']
TARGET = hypers['target']
NUM_MARKERS = hypers['num_markers']
if hypers['kernel_name'] == "GaussianKernel":
    KERNEL_TYPE = GaussianKernel
elif hypers['kernel_name'] == "LaplaceKernel":  
    KERNEL_TYPE = LaplaceKernel
elif hypers['kernel_name'] == "ReluNTK":
    KERNEL_TYPE = ReluNTK
else:
    raise ValueError(f"Unknown kernel {hypers['kernel_name']}")
KERNEL_WIDTH = hypers['kernel_width']
RIDGE = hypers['ridge']
FEWER_TRIALS = hypers['fewer_trials']

target2classes = {
    # plane car ship truck vs bird cat deer dog frog horse
    "vehicle": [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]],
    # cat dog horse vs bird deer frog
    "domesticated": [[3, 5, 7], [2, 4, 6]],
    "evenodd": [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]],
    "loops": [[0, 6, 8, 9], [1, 3, 5, 7]],
    "plane-frog": [[0], [6]],
    "car-ship": [[1], [8]],
    "bird-cat": [[2], [3]],
    "deer-horse": [[4], [7]],
    "dog-else": [[5], [0, 1, 2, 3, 4, 6, 7, 8, 9]],
    "cat-else": [[3], [0, 1, 2, 4, 5, 6, 7, 8, 9]],
    "dog-frog": [[5], [6]],
    "car-truck": [[1], [9]],
    "4-2": [[4], [2]],
    "4-9": [[4], [9]],
    "0-else": [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9]],
    "primes": [[2, 3, 5, 7], [4, 6, 8, 9]]
}


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
    X = ensure_torch(torch.normal(0, 1, (N_SAMPLES, DATA_DIM)))
    X *= torch.sqrt(ensure_torch(data_eigvals))
if DATASET in ["cifar5m", "imagenet32", "svhn"]:
    classes = target2classes.get(TARGET, None)
    X, labels = get_binarized_dataset(DATASET, classes, N_SAMPLES)
    X = ensure_torch(X)
    X = preprocess(X, center=True, normalize=NORMALIZE, zca_strength=ZCA_STRENGTH)
    # ensure typical sample has unit norm
    S = torch.linalg.svdvals(X)
    X *= torch.sqrt(N_SAMPLES / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()

d_eff = 1/(data_eigvals**2).sum()
print(f"d_eff: {d_eff:.2f}", end="\n")

eval_level_coeff = KERNEL_TYPE.get_level_coeff_fn(data_eigvals=data_eigvals,
                                                  kernel_width=KERNEL_WIDTH)
print("Generating hermites...", end=" ")
hea_eigvals, monomials = generate_hea_monomials(data_eigvals, P_MODES, eval_level_coeff)
H = compute_hermite_basis(X, monomials)
print("done.")

targets = {}
if TARGET == "monomials":
    monomial_idxs = get_monomial_targets(monomials, hea_eigvals, n_markers=NUM_MARKERS)
    print(f"Total target monomials: {len(monomial_idxs)}")
    for idx in monomial_idxs:
        ystar = ensure_numpy(H[:, idx])
        # ensure size(y_i) ~ 1
        targets[idx] = np.sqrt(N_SAMPLES) * ystar / np.linalg.norm(ystar)    
elif TARGET == "powerlaws":
    source_exps = [1.05, 1.1, 1.15, 1.25, 1.4, 1.6]
    for source_exp in source_exps:
        powerlaw = get_powerlaw(P_MODES, source_exp, offset=6)
        vstar = np.random.normal(0, 1, (P_MODES,)) * np.sqrt(powerlaw)
        vstar[0] = 0 # no constant mode
        vstar /= np.linalg.norm(vstar)
        y = (H / np.linalg.norm(H, axis=0, keepdims=True)) @ vstar
        # ensure size(y_i) ~ 1
        ystar = np.sqrt(N_SAMPLES) * y / np.linalg.norm(y)
        tail = P_MODES**(1-source_exp)
        noise = np.random.normal(0, 1, size=N_SAMPLES)
        ystar = np.sqrt(1-tail) * ystar + np.sqrt(tail) * noise
        targets[source_exp] = ystar
elif TARGET in target2classes:
    targets[TARGET] = labels

print("Starting Gram-Schmidt...", end=" ")
Q, _ = torch.linalg.qr(ensure_torch(H))
Q = ensure_numpy(Q)
torch.cuda.empty_cache()
print("done.")

print("Computing kernel matrix...", end=" ")
X_krn = X[:N_KERNEL, :]
kernel = KERNEL_TYPE(X_krn, kernel_width=KERNEL_WIDTH)
K = ensure_torch(kernel.K)
torch.cuda.empty_cache()
print("done.")


# START EXPERIMENT
##################

print()

def get_ntrials(ntrain):
    if FEWER_TRIALS:
        return max(1, min(20, N_KERNEL // ntrain))
    if ntrain < 100: return 50
    elif ntrain < 1000: return 20
    return max(5, N_KERNEL // ntrain)

ntrains = np.logspace(1, np.log10(N_TRAIN_MAX), base=10, num=30).astype(int)

et_mse = ExptTrace(["trial", "ntrain", "target"])
for target, ystar in targets.items():
    print("Starting target: ", target)
    ystar = ensure_torch(ystar[:N_KERNEL])
    print(f"ntrains:", end=" ", flush=True)
    for ntrain in ntrains:
        print(f"{ntrain}", end=" ", flush=True)
        ntrials = get_ntrials(ntrain)
        test_mse, _ = krr_loop(K, ystar, ntrain, N_TEST, ntrials, ridge=RIDGE)
        for trial, mse in enumerate(test_mse):
            et_mse[trial, ntrain, target] = mse
    print()
torch.cuda.empty_cache()


# CLEANUP
##################

print("diagonalizing kernel...", end=" ")
emp_eigvals, emp_eigvecs = kernel.eigendecomp()
print("done.")

print("computing coeffs...", end=" ")
coeffs = {}
Q = ensure_torch(Q)
for target, ystar in targets.items():
    y = ensure_torch(ystar / np.linalg.norm(ystar))
    v_emp = emp_eigvecs.T @ y[:N_KERNEL] / torch.linalg.norm(y[:N_KERNEL])
    v_hat = Q.T @ y / torch.linalg.norm(y)
    coeffs[target] = (ensure_numpy(v_hat), ensure_numpy(v_emp))
print("done.")

print("saving results...", end=" ")
expt_fm.save(targets, "targets.pickle")
expt_fm.save(coeffs, "coeffs.pickle")
result = {
    "monomials": [dict(m) for m in monomials],
    "d_eff": d_eff,
    "emp_eigvals": ensure_numpy(emp_eigvals),
    "th_eigvals": hea_eigvals,
    "mse": et_mse.serialize()
}
expt_fm.save(result, "result.pickle")
torch.cuda.empty_cache()
print("done.")
