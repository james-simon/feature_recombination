import numpy as np

import torch
from torch.multiprocessing import get_context
import torch.multiprocessing as mp
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
MAX_ITER = int(1e4)
EMA_SMOOTHER = 0.9

trial_counts = np.array([trial_count_fn(n) for n in NS], dtype=int)
max_trials   = int(trial_counts.max())

RNG = np.random.default_rng(SEED)
GEN = torch.Generator().manual_seed(SEED)
np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

## --- Dataset-specific ---
if DATASET == "synthetic":
    from data import get_synthetic_dataset
    DIM = 3072
    
    OFFSET=3
    ALPHA=2.01
    NOISE_SIZE=1
    YOFFSET=1.2
    BETA=1.2

    X_full, y_full, H, monomials, fra_eigvals, _, data_eigvals = get_synthetic_dataset(d=DIM, N=N_TOT, offset=OFFSET, alpha=ALPHA, cutoff_mode=CUTOFF_MODE,
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


U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)

## --- Target function defs ---
if TARGET_FUNCTION_TYPE == "monomial":
    from data import monomial_batch_fn
    target_monomials = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3}),] #Monomial({1:1, 3:1, 4:1})]
    
    total_iters = int(len(target_monomials) * trial_counts.sum())
    
    batch_function = lambda target_monomial, n, X, y: monomial_batch_fn(lambdas, Vt, target_monomial, dim, n, data_eigvals, N_TOT, gen=GEN, X=X, y=y)

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

    total_iters = int(len(source_exps) * trial_counts.sum())
    batch_function = lambda source_exponent, n: powerlaw_batch_fn(H, source_exponent)
    targets = source_exps
    locs = torch.arange(len(source_exps))
else:
    pass

dim = X_full.shape[1]

var_axes = ["target", "ntrain", "trial"]
et_losses = ExptTrace(var_axes)

## --- Multiprocessing setup ---

NUM_GPUS = torch.cuda.device_count()

def run_job(device_id, job, shared_config):
    """
    job: (target, n, trial)
    shared_config: anything read-only you want to avoid capturing from globals
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    torch.set_num_threads(1)  # avoid CPU contention when many procs

    target, n, trial = job
    X_te, y_te = batch_function(target, shared_config["N_TEST"], X=None, y=None)(0)
    
    X_tr, y_tr = batch_function(target, n, X=None, y=None)(trial)
    bfn = batch_function(target, n, X=X_tr, y=y_tr)
    
    model = MLP(d_in=shared_config["DIM"], depth=shared_config["DEPTH"],
                d_out=1, width=shared_config["WIDTH"]).to(device)

    outdict = train_network(
        model=model,
        batch_function=bfn,
        lr=shared_config["LR"],
        max_iter=shared_config["MAX_ITER"],
        percent_thresholds=shared_config["PERCENT_THRESHOLDS"],
        gamma=shared_config["GAMMA"],
        ema_smoother=shared_config["EMA_SMOOTHER"],
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_te, y_te=y_te,
    )

    test_losses = outdict["test_losses"]

    # Cleanup GPU memory
    del outdict, model, X_tr, y_tr, X_te, y_te
    torch.cuda.empty_cache()
    gc.collect()

    return (n, str(target), int(trial), test_losses)


def worker(device_id, job_queue, result_queue, shared_config):
    torch.cuda.set_device(device_id)
    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            key_n, key_t, key_trial, test_losses = run_job(device_id, job, shared_config)
            result_queue.put(("ok", (key_n, key_t, key_trial,
                                     torch.as_tensor(test_losses).cpu().numpy())))
        except Exception as e:
            result_queue.put(("err", (job, repr(e))))

shared_config = dict(
    DIM=DIM, DEPTH=DEPTH, WIDTH=WIDTH, LR=LR, GAMMA=GAMMA,
    EMA_SMOOTHER=EMA_SMOOTHER, MAX_ITER=MAX_ITER,
    PERCENT_THRESHOLDS=PERCENT_THRESHOLDS, N_TEST=N_TEST,
)

## --- Multiprocessing execution ---
def main():
    jobs = [(target, n, trial)
            for target in targets
            for nidx, n in enumerate(NS)
            for trial in range(int(trial_counts[nidx]))]

    ctx = get_context("spawn")

    job_queue    = ctx.Queue()
    result_queue = ctx.Queue()

    for job in jobs:
        job_queue.put(job)
    for _ in range(NUM_GPUS):
        job_queue.put(None)

    procs = [ctx.Process(target=worker, args=(dev, job_queue, result_queue, shared_config))
            for dev in range(NUM_GPUS)]
    for p in procs: p.start()

    total = len(jobs)
    done = 0
    
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        while done < total:
            kind, payload = result_queue.get()
            if kind == "ok":
                n, tstr, trial, test_losses = payload
                et_losses[n, tstr, trial] = test_losses
            else:
                job, err = payload
                print(f"[ERROR] {job}: {err}")
            done += 1
            pbar.update(1)

    for p in procs: p.join()

    result = {
        "targets": targets,
        "fra_eigvals": fra_eigvals.cpu(),
        "locs": locs.cpu(),
        "losses": et_losses.serialize()
    }

    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()

# # --- Trainloop ---
# pbar = tqdm(total=total_iters, desc="Processing items")
# for idx, target in enumerate(targets):
#     X_te, y_te = batch_function(target, N_TEST)(0)
#     for nidx, n in enumerate(NS):
#         num_trials_n = int(trial_counts[nidx])
#         for trial in range(num_trials_n):
#             pbar.set_postfix(
#                 current_item=f"Target:{target} n:{n} trial:{trial+1}/{num_trials_n}",
#                 refresh=False
#             )
#             bfn = batch_function(target, n)
#             X_tr, y_tr = bfn(trial)
#             model = MLP(d_in=dim, depth=DEPTH, d_out=1, width=WIDTH).to(device)

#             outdict = train_network(model, bfn, lr=LR, max_iter=MAX_ITER, percent_thresholds=PERCENT_THRESHOLDS,
#                                     gamma=GAMMA, ema_smoother=EMA_SMOOTHER, X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te)

#             # all_losses[nidx, idx, trial] = outdict["test_losses"]
#             # pct_breakpoints[nidx, idx, :, trial] = outdict["timekeys"]
#             et_losses[str(target), n, trial] = outdict["test_losses"]

#             del outdict, model, X_tr, y_tr
#             gc.collect()

#             pbar.update(1)

# pbar.close()

## --- Save results ---

if __name__ == "__main__":

    ## --- Setup file management ---

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

    mp.set_start_method("spawn", force=True)
    main()