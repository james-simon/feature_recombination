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
from utils import ensure_torch, derive_seed, seed_everything
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
NS = np.logspace(1, 3, 20, dtype=int)
DATASET = "synthetic"
kerneltype = GaussianKernel
TARGET_FUNCTION_TYPE = "monomial" # monomial, powerlaw, etc.
ONLYTHRESHOLDS = True # if True, only record last loss instead of full curve

SEED = 42

# MLP HPs
LR = 1e-0
DEPTH = 2
WIDTH = 8192
GAMMA = 1

# Training HPs
PERCENT_THRESHOLDS = (0.5, 1e-12) #needs a len atm
MAX_ITER = int(1e20)
EMA_SMOOTHER = 0.9
DETERMINSITIC = True
trial_counts = np.array([trial_count_fn(n) for n in NS], dtype=int)
max_trials   = int(trial_counts.max())

global_config = dict(DEPTH=DEPTH, WIDTH=WIDTH, LR=LR, GAMMA=GAMMA,
    EMA_SMOOTHER=EMA_SMOOTHER, MAX_ITER=MAX_ITER,
    PERCENT_THRESHOLDS=PERCENT_THRESHOLDS, N_TEST=N_TEST,
    SEED=SEED, ONLYTHRESHOLDS=ONLYTHRESHOLDS,
)

# Dataset HPs
# Note: not all of these are used for all datasets
datasethps = {"normalized": True,
              "cutoff_mode": 40_000,
              "d": 3072,
              "offset": 3,
              "alpha": 1.14,
              "noise_size": 1,
              "yoffset": 1.2,
              "beta": 1.2,
              "classes": None,
              "binarize": False}


## ----- Technicals below -----



## --- Multiprocessing setup ---

NUM_GPUS = torch.cuda.device_count()

def run_job(device_id, job, global_config, bfn_config=None):
    """
    job: (target, n, trial)
    global_config: anything read-only you want to avoid capturing from globals
    """
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    base_seed = global_config.get("SEED", None)
    job_seed  = derive_seed(base_seed, device_id)
    GEN, RNG = seed_everything(job_seed, device_id)

    torch.set_num_threads(1)  # avoid CPU contention when many procs

    if TARGET_FUNCTION_TYPE == "monomial":
        from data import monomial_batch_fn
        batch_function = lambda target_monomial, n, X, y: monomial_batch_fn(**bfn_config, monomial=target_monomial, bsz=n, gen=GEN, X=X, y=y)
    elif TARGET_FUNCTION_TYPE == "powerlaw":
        from data import powerlaw_batch_fn
        batch_function = lambda source_exponent, n, X, y: powerlaw_batch_fn(**bfn_config, source_exp=source_exponent, X=X, y=y, device=device)
    target, n, trial = job
    X_te, y_te = batch_function(target, global_config["N_TEST"], X=None, y=None)(0)
    
    X_tr, y_tr = batch_function(target, n, X=None, y=None)(trial)
    bfn = batch_function(target, n, X=X_tr, y=y_tr)
    
    model = MLP(d_in=global_config["DIM"], depth=global_config["DEPTH"],
                d_out=1, width=global_config["WIDTH"]).to(device)

    outdict = train_network(
        model=model,
        batch_function=bfn,
        lr=global_config["LR"],
        max_iter=global_config["MAX_ITER"],
        percent_thresholds=global_config["PERCENT_THRESHOLDS"],
        gamma=global_config["GAMMA"],
        ema_smoother=global_config["EMA_SMOOTHER"],
        only_thresholds=global_config["ONLYTHRESHOLDS"],
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_te, y_te=y_te,
    )

    train_losses = outdict["train_losses"]
    test_losses = outdict["test_losses"]

    # Cleanup GPU memory
    del outdict, model, X_tr, y_tr, X_te, y_te
    torch.cuda.empty_cache()
    gc.collect()

    return (n, str(target), int(trial), train_losses, test_losses)


def worker(device_id, job_queue, result_queue, global_config, bfn_config):
    torch.cuda.set_device(device_id)

    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            key_n, key_t, key_trial, train_losses, test_losses = run_job(device_id, job, global_config, bfn_config)
            result_queue.put(("ok", (key_n, key_t, key_trial,
                                     torch.as_tensor(train_losses).cpu().numpy(),
                                     torch.as_tensor(test_losses).cpu().numpy())))
        except Exception as e:
            result_queue.put(("err", (job, repr(e))))

## --- Multiprocessing execution ---
def main():
    ## --- Grab targets and such ---
    if DATASET == "synthetic":
        from data import get_synthetic_dataset

        X_full, _, H, monomials, fra_eigvals, _, data_eigvals = get_synthetic_dataset(**datasethps, N=N_TOT, kerneltype=kerneltype,
                                                                                           gen=torch.Generator(device='cuda').manual_seed(SEED))

    elif DATASET == "cifar10":
        #todo: get monomials
        from imdata import ImageData
        PIXEL_NORMALIZED =  False
        classes = datasethps['classes']
        normalized = datasethps['normalized']

        if classes is not None:
            imdata = ImageData('cifar10', "../data", classes=classes, onehot=len(classes)!=2, format="N")
        else:
            imdata = ImageData('cifar10', "../data", classes=classes, onehot=False, format="N")
        X_train, y_train = imdata.get_dataset(N_TRAIN, **datasethps, get='train',
                                            centered=True, normalize=PIXEL_NORMALIZED)
        X_test, y_test = imdata.get_dataset(N_TEST, **datasethps, get='test',
                                            centered=True, normalize=PIXEL_NORMALIZED)
        X_train, y_train, X_test, y_test = map(ensure_torch, (X_train, y_train, X_test, y_test))
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        X_full = torch.cat((X_train, X_test), dim=0)
        y_full = torch.cat((y_train, y_test), dim=0)
        X_train, y_train, X_test, y_test = [t/torch.linalg.norm(t) for t in (X_train, y_train, X_test, y_test)] if normalized else (X_train, y_train, X_test, y_test)
        data_eigvals = torch.linalg.svdvals(X_full)**2


    U, lambdas, Vt = torch.linalg.svd(X_full, full_matrices=False)
    dim = X_full.shape[1]

    ## --- Target function defs ---
    if TARGET_FUNCTION_TYPE == "monomial":
        target_monomials = [Monomial({10: 1}), Monomial({190:1}), Monomial({0:2}), Monomial({2:1, 3:1}), Monomial({15:1, 20:1}), Monomial({0:3}),] #Monomial({1:1, 3:1, 4:1})]
        
        total_iters = int(len(target_monomials) * trial_counts.sum())

        targets = target_monomials
        
        locs = torch.zeros(len(target_monomials))

        for i, monomial in enumerate(target_monomials):
            loc = np.where(np.array(monomials) == monomial)[0][0]
            locs[i] = loc
        bfn_config = dict(lambdas=lambdas, Vt=Vt, dim=dim, data_eigvals=data_eigvals, N=N_TOT)

    elif TARGET_FUNCTION_TYPE == "powerlaw":
        source_exps = [1.01, 1.1, 1.25, 1.5, 2.0]
        from data import get_powerlaw
        classes = datasethps['classes']
        cutoff_mode = datasethps['cutoff_mode']
        normalized = datasethps['normalized']
        offset = datasethps['yoffset']
        squared_coeffs = get_powerlaw(cutoff_mode, source_exps, offset, normalized=normalized)
    

        total_iters = int(len(source_exps) * trial_counts.sum())
        targets = source_exps
        locs = torch.arange(len(source_exps))
        bfn_config = dict(H=H)
    
    global_config.update(dict(DIM=dim))

    var_axes = ["target", "ntrain", "trial"]
    et_losses = ExptTrace(var_axes)
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

    procs = [ctx.Process(target=worker, args=(dev, job_queue, result_queue, global_config, bfn_config))
            for dev in range(NUM_GPUS)]
    for p in procs: p.start()

    total = len(jobs)
    done = 0
    
    with tqdm(total=total, desc="Runs", dynamic_ncols=True) as pbar:
        while done < total:
            kind, payload = result_queue.get()
            if kind == "ok":
                n, tstr, trial, train_losses, test_losses = payload
                et_losses[n, tstr, trial] = test_losses
                if not(ONLYTHRESHOLDS):
                    train_losses = train_losses[-1]
                    test_losses = test_losses[-1]
                pbar.set_postfix_str(
                f"train {train_losses:.3g} | test {test_losses:.3g} | n={n} | target={tstr} | trial={trial}",
                refresh=False
            )
            else:
                job, err = payload
                print(f"[ERROR] {job}: {err}")
            done += 1
            pbar.update(1)

    for p in procs: p.join()

    result = {
        "targets": targets,
        "monomials": monomials,
        "fra_eigvals": fra_eigvals.cpu(),
        "locs": locs.cpu(),
        "losses": et_losses.serialize()
    }

    expt_fm.save(result, "result.pickle")
    torch.cuda.empty_cache()

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
                width=WIDTH, depth=DEPTH, gamma=GAMMA, lr=LR, seed=SEED, n_tot=N_TOT, n_test=N_TEST)
    with open(expt_fm.get_filename("hypers.json"), 'w') as f:
        json.dump(hypers, f, indent=4)

    ## --- Begin experiment ---
    mp.set_start_method("spawn", force=True)
    main()