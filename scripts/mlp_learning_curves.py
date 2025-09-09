import numpy as np

import torch
from tqdm import tqdm
import gc

import sys, os
import json

sys.path.append("../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from kernels import GaussianKernel
from feature_decomp import Monomial
from utils import ensure_torch
from mlps import MLP, train_network
from tools import trial_count_fn

from ExptTrace import ExptTrace
from FileManager import FileManager

colors = ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]
markers = ['x', 's', 'o', '^', 'D', '*', 'v', 'p', 'h']

## --- Experiment parameters ---

EXPT_NAME = "mlp-learning-curves"
N_TRAIN = 4_000
N_TEST = 1_000
N_TOT = N_TEST+N_TRAIN
NS = np.logspace(0, 3.3, 10, dtype=int)
DATASET = "synthetic"
CUTOFF_MODE = 40_000 #aka P
kerneltype = GaussianKernel
TARGET_FUNCTION_TYPE = "monomial" # monomial, powerlaw, etc.
NORMALIZED=True
SEED = 42

LR = 1e-2
DEPTH = 2
WIDTH = 8192
GAMMA = 1

PERCENT_THRESHOLDS = (0.75, 0.0001)
max_iter = int(1e4)
ema_smoother = 0.9

trial_counts = np.array([trial_count_fn(n) for n in NS], dtype=int)
max_trials   = int(trial_counts.max())

RNG = np.random.default_rng(SEED)
GEN = torch.Generator().manual_seed(SEED)

## --- Dataset-specific ---
if DATASET == "synthetic":
    from data import get_synthetic_dataset
    DIM = 3072
    
    OFFSET=3
    ALPHA=2.01
    NOISE_SIZE=1
    YOFFSET=1.2
    BETA=1.2

    X_full, y_full, H, monomials, fra_eigvals, data_eigvals = get_synthetic_dataset(d=DIM, N=N_TOT, offset=OFFSET, alpha=ALPHA, cutoff_mode=CUTOFF_MODE,
                                                                         noise_size=NOISE_SIZE, yoffset=YOFFSET, beta=BETA, normalized=NORMALIZED, kerneltype=kerneltype,
                                                                         gen=GEN)

elif DATASET == "cifar10":
    from imdata import ImageData

    CLASSES = None
    BINARIZED = False
    PIXEL_NORMALIZED = False

    if CLASSES is not None:
        imdata = ImageData('cifar10', "../data", classes=CLASSES, onehot=len(CLASSES)!=2, format="N")
    else:
        imdata = ImageData('cifar10', "../data", classes=CLASSES, onehot=False, format="N")
    X_train, y_train = imdata.get_dataset(N_TRAIN, get='train', rng=RNG, binarize=BINARIZED,
                                          centered=True, normalize=PIXEL_NORMALIZED)
    X_test, y_test = imdata.get_dataset(N_TEST, get='test', rng=RNG, binarize=BINARIZED,
                                          centered=True, normalize=PIXEL_NORMALIZED)
    X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    X_full = torch.cat((X_train, X_test), dim=0)
    y_full = torch.cat((y_train, y_test), dim=0)
    X_train, y_train, X_test, y_test = [t/torch.linalg.norm(t) for t in (X_train, y_train, X_test, y_test)] if NORMALIZED else (X_train, y_train, X_test, y_test)
    data_eigvals = torch.linalg.svdvals(X_full)**2

## --- Target function defs ---
if TARGET_FUNCTION_TYPE == "monomial":
    from data import monomial_batch_fn
    target_monomials = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3})]
    
    # all_losses = np.full((len(NS), len(target_monomials), max_trials, max_iter), np.nan, dtype=float)
    # pct_breakpoints = np.full((len(NS), len(target_monomials), len(PERCENT_THRESHOLDS), max_trials), np.nan, dtype=float)
    total_iters = int(len(target_monomials) * trial_counts.sum())
    batch_functions = lambda target_monomial, n: monomial_batch_fn(lambdas, Vt, target_monomial, dim, n, data_eigvals, N_TOT, gen=GEN)

    targets = target_monomials
    
    locs = torch.zeros(len(target_monomials))

    for i, monomial in enumerate(target_monomials):
        loc = np.where(np.array(monomials) == monomial)[0][0]
        locs[i] = loc

elif TARGET_FUNCTION_TYPE == "powerlaw":
    from data import powerlaw_batch_fn
    source_exps = [1.01, 1.1, 1.25, 1.5, 2.0]
    from data import get_powerlaw
    squared_coeffs = get_powerlaw(CUTOFF_MODE, source_exps, OFFSET, normalized=NORMALIZED)

    # all_losses = np.full((len(NS), len(source_exps), max_trials, max_iter), np.nan, dtype=float)
    # pct_breakpoints = np.full((len(NS), len(source_exps), len(PERCENT_THRESHOLDS), max_trials), np.nan, dtype=float)
    total_iters = int(len(source_exps) * trial_counts.sum())
    batch_functions = lambda source_exponent, n: powerlaw_batch_fn(H, source_exponent)
    targets = source_exps
    locs = torch.arange(len(source_exps))
else:
    pass

U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)

pbar = tqdm(total=total_iters, desc="Processing items")

dim = X_full.shape[0]

var_axes = ["ntrain", "target", "trial"]
et_losses = ExptTrace(var_axes)

## --- SETUP FILE MANAGEMENT ---

datapath = os.getenv("DATASETPATH")
exptpath = os.getenv("EXPTPATH")
if datapath is None:
    raise ValueError("must set $DATASETPATH environment variable")
if exptpath is None:
    raise ValueError("must set $EXPTPATH environment variable")
fp = f"MLPtraining-gamma:{GAMMA}-target:{TARGET_FUNCTION_TYPE}"
expt_dir = os.path.join(exptpath, "phlab", EXPT_NAME, DATASET, fp)

if not os.path.exists(expt_dir):
    os.makedirs(expt_dir)
expt_fm = FileManager(expt_dir)
print(f"Working in directory {expt_dir}.")
hypers = dict(expt_name=EXPT_NAME, dataset=DATASET, target_function_type=TARGET_FUNCTION_TYPE,
              width=WIDTH, depth=DEPTH, gamma=GAMMA, lr=LR, seed=SEED, n_tot=N_TOT, n_test=N_TEST, )
with open(expt_fm.get_filename("hypers.json"), 'w') as f:
    json.dump(hypers, f, indent=4)


# --- Trainloop ---

for idx, target in enumerate(targets):
    batch_function = batch_functions[idx]
    X_te, y_te = batch_function(target, N_TEST)
    for nidx, n in enumerate(NS):
        num_trials_n = int(trial_counts[nidx])

        for trial in range(num_trials_n):
            pbar.set_postfix(
                current_item=f"Target:{target} n:{n} trial:{trial+1}/{num_trials_n}",
                refresh=False
            )
            X_tr, y_tr = batch_function(target, n)
            model = MLP(d_in=dim, depth=DEPTH, d_out=1, width=WIDTH).to(device)

            outdict = train_network(model, batch_function, lr=LR, max_iter=max_iter, PERCENT_THRESHOLDS=PERCENT_THRESHOLDS,
                                    gamma=GAMMA, ema_smoother=ema_smoother, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

            # all_losses[nidx, idx, trial] = outdict["test_losses"]
            # pct_breakpoints[nidx, idx, :, trial] = outdict["timekeys"]
            et_losses[n, target, trial] = outdict["test_losses"]

            del outdict, model, X_tr, y_tr
            gc.collect()

            pbar.update(1)

pbar.close()

## --- Save results ---

result = {
    "targets": targets,
    "fra_eigvals": fra_eigvals.cpu(),
    "locs": locs.cpu(),
    "losses": et_losses.serialize()
}

expt_fm.save(result, "result.pickle")
torch.cuda.empty_cache()