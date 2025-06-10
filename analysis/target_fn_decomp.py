import numpy as np
import torch
from itertools import product

from utils import ensure_torch, get_matrix_hermites, get_standard_tools, ensure_numpy
from data import get_synthetic_X
from feature_decomp import generate_fra_monomials
from kernels import GaussianKernel
from utils import find_iterables, find_statics

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

def get_v_true(fra_eigvals, ytype='Gaussian', **kwargs):
    assert ytype in ["Gaussian", "Uniform", "Binarized", "PowerLaw", "OneHot", "NHot"], "Type not found"
    match ytype:
        case "Gaussian":
            return ensure_torch(torch.normal(fra_eigvals, torch.sqrt(fra_eigvals)))
        case "Uniform":
            return ensure_torch(torch.rand(fra_eigvals.shape))
        case "Binarized":
            H = kwargs.get("H", None)
            y_underlying = ensure_torch(torch.randint(low=0, high=2, size=(H.shape[0],)) * 2 - 1)
            v = torch.linalg.lstsq(H, y_underlying).solution
            return ensure_torch(v)
        case "PowerLaw":
            H = kwargs.get("H", None)
            i0 = kwargs.get("vi0", 3)
            alpha = kwargs.get("valpha", 1.5)
            pldim = H.shape[1]
            # pldim = kwargs.get("pldim", 400)
            pldecay = ensure_torch((i0+np.arange(pldim)) ** -alpha)
            pldecay /= torch.sqrt((pldecay**2).sum())
            v = pldecay
            # y_underlying = ensure_torch(torch.randint(low=0, high=2, size=(H.shape[0],)) * 2 - 1)
            # y_refactor = y_underlying - H[:, :pldim] @ pldecay
            # v_non_pl = torch.linalg.lstsq(H[:, pldim:], y_refactor).solution
            # v = torch.hstack((pldecay, v_non_pl)).T
            return ensure_torch(v)
        case "OneHot":
            H = kwargs.get("H", None)
            ohindex = kwargs.get("OneHotIndex", None)
            mode_dim = H.shape[1]
            if ohindex == None:
                ohindex = np.random.choice(mode_dim, replace=False)
            v = torch.zeros(mode_dim)
            v[ohindex] = 1
            return ensure_torch(v)
        case "NHot":
            indices = kwargs.get("NHotIndices", 3)
            nhsizes = kwargs.get("NHotSizes", None)
            H = kwargs.get("H", None)
            mode_dim = H.shape[1]
            if type(indices) == int: #if num_indices provided as opposed to the locations of the indices
                indices = np.random.choice(mode_dim, size=indices, replace=False)
            if nhsizes is None:
                nhsizes =  torch.tensor((1.2+np.arange(len(indices))) ** -3, dtype=torch.float32)
                nhsizes /= torch.sqrt((nhsizes**2).sum())
            v = torch.zeros(mode_dim, dtype=torch.float32)
            v[indices] = nhsizes
            return ensure_torch(v)
            
    return None

def get_synthetic_dataset(X=None, data_eigvals=None, ytype="Gaussian", d=500, N=15000, offset=3, alpha=1.5, cutoff_mode=10000,
                          noise_size=0.1, normalized=True, **vargs):
    """
    y_type: One of \"Gaussian\", \"Uniform\", \"Binarized\", "\PowerLaw\", \"OneHot\", \"NHot\"
    """
    if X is None:
        X, data_eigvals = get_synthetic_X(d=d, N=N, offset=offset, alpha=alpha)

    kernel_width = vargs.get("kernel_width", 2)
    kerneltype = vargs.get("kerneltype", GaussianKernel)
    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, cutoff_mode, kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals), kmax=9)
    H = ensure_torch(get_matrix_hermites(X, monomials))
    fra_eigvals = ensure_torch(fra_eigvals)
    v_true = get_v_true(fra_eigvals, ytype, noise_size=noise_size, H=H, **vargs)
    v_true = v_true if not normalized else v_true/torch.norm(v_true)
    y = ensure_torch(H) @ v_true + ensure_torch(torch.normal(0., noise_size, (H.shape[0],)))
    return X, y, H, monomials, fra_eigvals, v_true

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
    do_multiple_sampling = np.any(key in keys for key in ["d", "n_train", "offset", "alpha", "cutoff_mode", "noise_size", "normalized"])
    if not do_multiple_sampling or "H" not in static_dict:
        X, y, H, monomials, fra_eigvals, v_true = get_synthetic_dataset(**static_dict)
        static_dict.update(dict(X=X, y=y, H=H, monomials=monomials, fra_eigvals=fra_eigvals, v_true=v_true))

    for idx, combo in enumerate(product(*values)):
        combo_dict = dict(zip(keys, combo))
        all_args = {**static_dict, **combo_dict}
        print(f"Starting {combo_dict}")

        if do_multiple_sampling:
            print(f"Resampling")
            all_args.update({"X": None})
            del all_args["H"], all_args["fra_eigvals"]
            X, y, H, monomials, fra_eigvals, v_true = get_synthetic_dataset(**all_args)
            all_args.update(dict(X=X, y=y, H=H, monomials=monomials, fra_eigvals=fra_eigvals, v_true=v_true))

        out = sample_v_tilde(**all_args)

        multi_idx = np.unravel_index(idx, shapes)
        all_v_tildes[multi_idx] = out
    return all_v_tildes

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