import numpy as np
import torch
from itertools import product

from utils import ensure_torch, find_iterables, find_statics
from data import get_synthetic_dataset

def get_vtilde(H, y, method = "LSTSQ", **kwargs):
    """
    Assumes H is already made of floats
    """
    assert method in ["LSTSQ", "dotprod", "LSTSQR"], "method for finding eigencoeffs not found"
    #y_non_onehot = torch.argmax(y_train, dim=1)
    if method == "LSTSQ":
        v_tilde = torch.linalg.lstsq(ensure_torch(H), ensure_torch(y).unsqueeze(1)).solution.squeeze()
    elif method == "dotprod":
        v_tilde = (H.T @ y.float())
    elif method == "LSTSQR":
        H = ensure_torch(H).float()
        y = ensure_torch(y).float().unsqueeze(1)  # shape: [n, 1]
        n_features = H.shape[1]
        I = torch.eye(n_features, device=H.device, dtype=H.dtype)
        ridge = kwargs.get("ridge", 1)
        v_tilde = torch.linalg.solve(H.T @ H + ridge * I, H.T @ y).squeeze()

    return v_tilde

def sample_v_tilde(H=None, y=None, top_fra_eigmode=None, n_train=10, num_trials=20, method="LSTSQ", normalized=True, verbose_every=5, **kwargs):
    """
    Samples v_tilde by randomly selecting n_train samples from H and y.
    """
    Nmax = H.shape[0]
    norm_amount = np.sqrt(n_train) if normalized else 1
    v_tildes = torch.zeros(top_fra_eigmode, num_trials)
    for trial_idx in range(num_trials):
        if verbose_every is not None and not trial_idx%verbose_every:
            print(f"Starting run {trial_idx}")
        random_sampling = np.random.choice(Nmax, size=n_train, replace=False)
        v_tilde = get_vtilde(H[random_sampling, :top_fra_eigmode], y[random_sampling]/norm_amount, method=method, **kwargs)
        v_tildes[:, trial_idx] = v_tilde
    return v_tildes

def v_tilde_experiment(input_dict):
    #assumes all v_tildes will be similarly shaped
    iterable_dict = find_iterables(input_dict)
    static_dict = find_statics(input_dict)
    keys = list(iterable_dict.keys())
    values = list(iterable_dict.values())
    shapes = [len(v) for v in values]

    num_trials = static_dict.get("num_trials", iterable_dict.get("num_trials")) #should usually be in num_trials
    P = int(static_dict.get("top_fra_eigmode", np.min(iterable_dict.get("top_fra_eigmode"))))

    all_v_tildes = torch.zeros(*shapes, P, num_trials)

    #check if only one dataset needs to be made and everything can be based off that
    #or if we need to remake the dataset every time
    do_multiple_sampling = np.any([key in keys for key in ["d", "offset", "alpha", "cutoff_mode", "noise_size", "normalized"]])
    if not do_multiple_sampling:
        X, y, H, monomials, fra_eigvals, v_true = get_synthetic_dataset(**static_dict)
        static_dict.update(dict(X=X, y=y, H=H, monomials=monomials, fra_eigvals=fra_eigvals, v_true=v_true))

    for idx, combo in enumerate(product(*values)):
        combo_dict = dict(zip(keys, combo))
        all_args = {**static_dict, **combo_dict}
        print(f"Starting {combo_dict}")

        if do_multiple_sampling:
            print(f"Resampling")
            all_args.update({"X": None, "H": None, "fra_eigvals": None}) # None-define for first pass
            del all_args["H"], all_args["fra_eigvals"] #w/o deleting, causes issues in get_synth_dataset
            X, y, H, monomials, fra_eigvals, v_true = get_synthetic_dataset(**all_args)
            all_args.update(dict(X=X, y=y, H=H, monomials=monomials, fra_eigvals=fra_eigvals, v_true=v_true))

        out = sample_v_tilde(**all_args)

        multi_idx = np.unravel_index(idx, shapes)
        all_v_tildes[multi_idx] = out[:P, :]
    return all_v_tildes, H, v_true, monomials

#explicit experiments

def v_tilde_p_experiment(top_fra_eigmodes, H, y, n_train, num_trials=20, minmode=True, method="LSTSQ", normalized=True, verbose_every=5, ridge=None):
    minval = np.min(top_fra_eigmodes)

    all_v_tildes = [] if not minmode else torch.zeros(len(top_fra_eigmodes), minval, num_trials)
    for i, top_fra_eigmode in enumerate(top_fra_eigmodes):
        top_fra_eigmode = int(top_fra_eigmode) #usually a np.int which prints exception
        print(f"Starting P={top_fra_eigmode}")
        v_tildes = sample_v_tilde(H, y, top_fra_eigmode, n_train, num_trials, normalized=normalized, verbose_every=5, method=method, ridge=None)
        if not minmode:
            all_v_tildes.append(v_tildes)
        else:
            all_v_tildes[i] = v_tildes[:minval]
    if minmode:
        return all_v_tildes.squeeze()
    return all_v_tildes

#not super useful
def get_y_recon(H, v_tilde, y,):
    y_pred = (ensure_torch(H) @ v_tilde).squeeze()
    # err = ((y-y_pred)**2).mean()
    if y.ndim >= 2:
        y = np.argmax(y, axis=1)
    y_non_onehot_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    y_pred_sorted = [y_pred_np[y_non_onehot_np == i] for i in np.unique(y_non_onehot_np)]
    return y_pred_sorted