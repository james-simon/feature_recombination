import numpy as np
import torch
from itertools import product

from utils import ensure_torch, find_iterables, find_statics
from tools import get_standard_tools
from data_old import get_synthetic_dataset, ImageData
from tools import find_beta

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

def sample_v_tilde(H=None, v_true=None, y=None, top_fra_eigmode=None, n=10, n_trials=20, method="LSTSQ", verbose_every=5, eigcoeff_normalized=False, **kwargs):
    """
    Samples v_tilde by randomly selecting n samples from H and y.
    """
    Nmax = H.shape[0]
    v_tildes = torch.zeros(top_fra_eigmode, n_trials)
    residuals = torch.zeros(n_trials)
    for trial_idx in range(n_trials):
        if verbose_every is not None and not trial_idx%verbose_every:
            print(f"Starting run {trial_idx}")
        random_sampling = np.random.choice(Nmax, size=n, replace=False)
        v_tilde = get_vtilde(H[random_sampling, :top_fra_eigmode], y[random_sampling], method=method, **kwargs)
        v_tilde = v_tilde/torch.linalg.norm(v_tilde) if eigcoeff_normalized else v_tilde
        v_tildes[:, trial_idx] = v_tilde
        if v_true is not None:
            err = H[:, :top_fra_eigmode] @ (v_tilde - v_true[:top_fra_eigmode]) + H[:, top_fra_eigmode:] @ v_true[top_fra_eigmode:]
        else:
            err = H[:, :top_fra_eigmode] @ v_tilde - y
        residuals[trial_idx] = (err**2).mean()
    return v_tildes, residuals

def get_eigencoeffs(X=None, y=None, dataset_name="cifar10", n_train=None, n_test=None, kerneltype=None, kernel_width=2, beta=None,
                    num_estimators=20, n_trials=20, n_trials_beta=10, rng=np.random.default_rng(42), kappa=None, ridge=None, P_optimal=None,
                    **dataargs):
    """
    From a dataset, estimates eigencoefficients
    
    Returns dict of eigencoefficients, optimal num eigenmodes considered, """

    if X is None:
        imdata = ImageData(dataset_name, **dataargs)
        
        X_train, y_train = imdata.get_dataset(n_train, get='train', rng=rng, **dataargs)
        X_test, y_test = imdata.get_dataset(n_test, get='test', rng=rng, **dataargs)
        X_train, y_train, X_test, y_test = [ensure_torch(t) for t in (X_train, y_train, X_test, y_test)]
        X = torch.vstack((X_train, X_test))
        y = torch.vstack((y_train, y_test)).squeeze()
    
    n_tot = n_train+n_test
    assert n_tot == len(y), "Error found while evaluating the number of samples in the dataset"
    
    monomials, kernel, H, fra_eigvals, data_eigvals = get_standard_tools(X, kerneltype, kernel_width, top_mode_idx = X.shape[0], data_eigvals = None, kmax=10)

    K = kernel.K
    intercept = None #force in case beta is given
    if P_optimal is None:
        if beta is None:
            beta, intercept = find_beta(K, y, num_estimators=num_estimators, n_test=n_test, n_trials=n_trials_beta)
            # print(f"Found beta = {beta}")
        P_optimal = int((beta-1)/beta*n_tot)
    eigencoeffs, train_mse = sample_v_tilde(H, y=y, top_fra_eigmode=P_optimal, n=n_tot, n_trials=n_trials, method="LSTSQ", verbose_every=None, eigcoeff_normalized=False)
    train_mse = train_mse[0]
    if kappa is None:
        test_mse = train_mse * (n_tot/(n_tot-P_optimal))**(2.)
    else:
        test_mse = train_mse * (n_tot*kappa/ridge)**(2.)
    
    noise_var = test_mse/n_tot
    good_indices = ((eigencoeffs.mean(axis=-1))**(2)/noise_var > 1)

    retdict = {"eigencoeffs": eigencoeffs, "P_optimal": P_optimal, "N":n_tot, "kernel": K, "test_mse": test_mse,
               "monomials": monomials, "H": H, "fra_eigvals": fra_eigvals, "data_eigvals": data_eigvals,
               "eigcoeff_var": noise_var, "beta": beta, "intercept": intercept, "resolveable_indices": good_indices}
    return retdict

def dirac_eigencoeffs(H=None, y=None, n=10, n_trials=20, method="LSTSQ", verbose_every=5, eigcoeff_normalized=False, **kwargs):
    """
    Try to estimate v_tilde by getting coefficients one at a time.
    """
    Nmax, num_eigvecs = H.shape
    v_tildes = torch.zeros(num_eigvecs, n_trials)
    for eigvec in range(num_eigvecs):
        for trial_idx in range(n_trials):
            if verbose_every is not None and not trial_idx%verbose_every:
                print(f"Starting run {trial_idx}")
            random_sampling = np.random.choice(Nmax, size=n, replace=False)
            v_tilde = get_vtilde(H[random_sampling, eigvec].unsqueeze(1), y[random_sampling], method=method, **kwargs)
            v_tilde = v_tilde/torch.linalg.norm(v_tilde) if eigcoeff_normalized else v_tilde
            v_tildes[:, trial_idx] = v_tilde
    return v_tildes

def v_tilde_experiment(input_dict):
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

        out = sample_v_tilde(**all_args)

        multi_idx = np.unravel_index(idx, shapes)
        all_v_tildes[multi_idx] = out[:P, :]
    return all_v_tildes, H, v_true, monomials

#explicit experiments

def v_tilde_p_experiment(top_fra_eigmodes, H, y, n, n_trials=20, minmode=True, method="LSTSQ", verbose_every=5, ridge=None, **kwargs):
    minval = np.min(top_fra_eigmodes)

    all_v_tildes = [] if not minmode else torch.zeros(len(top_fra_eigmodes), minval, n_trials)
    for i, top_fra_eigmode in enumerate(top_fra_eigmodes):
        top_fra_eigmode = int(top_fra_eigmode) #usually a np.int which prints exception
        print(f"Starting P={top_fra_eigmode}")
        v_tildes = sample_v_tilde(H, y, top_fra_eigmode, n, n_trials, verbose_every=5, method=method, ridge=None, **kwargs)
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