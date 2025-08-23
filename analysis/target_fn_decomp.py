import numpy as np
import torch
from itertools import product

from utils import ensure_torch
from data_old import get_synthetic_dataset
from tqdm import trange

#helper fns v_tilde_experiment
def find_iterables(d, no_list=["H", "y"]):
    return {k: v for k, v in d.items() if (isinstance(v, (list, np.ndarray)) and k not in no_list)}

def find_statics(d):
    return {k: v for k, v in d.items() if not isinstance(v, (list, np.ndarray))}

def get_eigencoeffs(H, y, method = "GRF", **kwargs):
    """
    Assumes H is already made of floats
    """
    assert method in ["LSTSQ", "dotprod", "LSTSQR", "GRF"], "method for finding eigencoeffs not found"
    #y_non_onehot = torch.argmax(y_train, dim=1)
    if method == "LSTSQ":
        coeffs = torch.linalg.lstsq(ensure_torch(H), ensure_torch(y).unsqueeze(1)).solution.squeeze()
        residual = y - H @ coeffs
        residual_norm_squared = torch.norm(residual) ** 2
    elif method == "dotprod":
        coeffs = (H.T @ y.float())
        residual = y - H @ coeffs
        residual_norm_squared = torch.norm(residual) ** 2
    elif method == "LSTSQR":
        H = ensure_torch(H).float()
        y = ensure_torch(y).float().unsqueeze(1)  # shape: [n, 1]
        n_features = H.shape[1]
        I = torch.eye(n_features, device=H.device, dtype=H.dtype)
        ridge = kwargs.get("ridge", 1)
        coeffs = torch.linalg.solve(H.T @ H + ridge * I, H.T @ y).squeeze()
        residual = y.squeeze() - H @ coeffs
        residual_norm_squared = torch.norm(residual) ** 2
    elif method == "GRF":
        n_steps = kwargs.get("n_steps", None)
        n_samples, n_modes = H.shape
        if n_steps is None:
            n_steps = n_modes

        assert n_steps <= n_modes, "n_steps must be less than or equal to n_modes"

        coeffs = torch.zeros(n_steps, dtype=H.dtype, device=H.device)
        residual = y.clone()

        with trange(n_steps, desc="Sequential fit", unit="step", total=n_steps) as pbar:
            for j in pbar:
                phi_j = H[:, j]
                coeffs[j] = torch.dot(phi_j, residual) / torch.linalg.norm(phi_j) ** 2
                residual -= coeffs[j] * phi_j
                pbar.set_postfix(residual_norm=residual.norm().item())
        residual_norm_squared = residual.norm().item() ** 2
    return coeffs, residual_norm_squared

def sample_eigencoeffs(H=None, v_true=None, y=None, top_fra_eigmode=None, n=10, n_trials=20, method="LSTSQ", verbose_every=5, eigcoeff_normalized=True, **kwargs):
    """
    Samples v_tilde by randomly selecting n samples from H and y.
    """
    Nmax = H.shape[0]
    v_tildes = torch.zeros(top_fra_eigmode, n_trials)
    residuals_squared = torch.zeros(n_trials)
    for trial_idx in range(n_trials):
        if verbose_every is not None and not trial_idx%verbose_every:
            print(f"Starting run {trial_idx}")
        random_sampling = np.random.choice(Nmax, size=n, replace=False)
        v_tilde, residual_squared = get_eigencoeffs(H[random_sampling, :top_fra_eigmode], y[random_sampling], method=method, **kwargs)
        v_tilde = v_tilde/torch.linalg.norm(v_tilde) if eigcoeff_normalized else v_tilde
        v_tildes[:, trial_idx] = v_tilde
        residuals_squared[trial_idx] = residual_squared
    return v_tildes, residuals_squared

def eigencoeff_experiment(input_dict):
    #assumes all v_tildes will be similarly shaped
    iterable_dict = find_iterables(input_dict)
    static_dict = find_statics(input_dict)
    keys = list(iterable_dict.keys())
    values = list(iterable_dict.values())
    shapes = [len(v) for v in values]

    n_trials = static_dict.get("n_trials", iterable_dict.get("n_trials")) #should usually be in n_trials
    P = int(static_dict.get("top_fra_eigmode", np.min(iterable_dict.get("top_fra_eigmode"))))

    all_v_tildes = torch.zeros(*shapes, P, n_trials)

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

        out = sample_eigencoeffs(**all_args)

        multi_idx = np.unravel_index(idx, shapes)
        all_v_tildes[multi_idx] = out[:P, :]
    return all_v_tildes, H, v_true, monomials