import numpy as np
import torch
from utils.general import ensure_torch
from utils.stats import get_standard_tools
from utils.hermite import get_matrix_hermites
from data import get_train_dataset

def get_eigenfunctions(monomials, n_train=50000, dataset_name="cifar10", classes=None, rng=np.random.default_rng(1)):
    X_train, y = get_train_dataset(n_train, dataset_name, classes=classes, rng=rng, center=True, normalize=False, binarize=True)
    H = get_matrix_hermites(X_train, monomials)
    return ensure_torch(H)

def get_vtilde(H, y, method = "LSTSQ", ridge = None):
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
        v_tilde = torch.linalg.solve(H.T @ H + ridge * I, H.T @ y).squeeze()

    return v_tilde

def get_y_recon(H, v_tilde, y, classes):
    y_pred = (ensure_torch(H) @ v_tilde).squeeze()
    # err = ((y-y_pred)**2).mean()
    y_non_onehot_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    y_pred_sorted = [y_pred_np[y_non_onehot_np == i] for i in range(len(classes))]
    return y_pred_sorted


# Plotting code
### Major WIP below - don't use!
### See google colab code for actual implementation

def get_all_vtildes(num_trials, n_trains, top_fra_eigmodes, rng, classes, kerneltype, kernel_width=2):
    all_v_tildes = []
    for n in n_trains:
        print(f"Starting N={n}")
        for p in top_fra_eigmodes:
            # indices = np.linspace(1, top_fra_eigmode+1, top_fra_eigmode)
        
            print(f"Starting P={p}")
            v_tildes = torch.zeros(p, num_trials)
            for trial_idx in range(num_trials):
                print(f"Starting run {trial_idx}")
                if trial_idx == 0:
                    X_train, y = get_train_dataset(n, dataset_name="cifar10", rng=rng, classes=classes, center=True, normalize=False, binarize=True)
                    monomials, _, H, _, _ = get_standard_tools(X_train, kerneltype, kernel_width, top_mode_idx=p)
                    v_tildes[:, trial_idx] = get_vtilde(H, y, method = "LSTSQ", ridge = None)
                else:
                    X_train, y = get_train_dataset(n, dataset_name="cifar10", rng=rng, classes=classes, center=True, normalize=False, binarize=True)
                    H = ensure_torch(get_matrix_hermites(X_train, monomials))
                    v_tildes[:, trial_idx] = get_vtilde(H, y, method = "LSTSQ", ridge = None)
                    
            all_v_tildes.append(v_tildes)

def plot_vtilde(ax, v_tildes, monomials, errorbars, colors, title):
    indices = np.linspace(1, len(v_tildes)+1, len(v_tildes))
        
    mean_v = v_tildes.abs().mean(axis=1)
    std_v = v_tildes.abs().std(axis=1)
    degrees = np.array([monomial.degree() for monomial in monomials])
    for degree in np.unique(degrees):
        idxs = np.where(np.array(degrees) == degree)[0]
        if errorbars:
            ax.errorbar(indices[idxs], mean_v[idxs], yerr=std_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1,)
        else:
            ax.scatter(indices[idxs], mean_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1,)
        
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")