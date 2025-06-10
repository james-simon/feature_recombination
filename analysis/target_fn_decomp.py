import numpy as np
import torch
from utils import ensure_torch, get_matrix_hermites, get_standard_tools, ensure_numpy
# from data import get_train_dataset
import matplotlib.pyplot as plt
from data import get_synthetic_X
from feature_decomp import generate_fra_monomials
from kernels import GaussianKernel

#get_train_dataset not defined yet?
# def get_eigenfunctions(monomials, n_train=50000, dataset_name="cifar10", classes=None, rng=np.random.default_rng(1)):
#     X_train, y = get_train_dataset(n_train, dataset_name, classes=classes, rng=rng, center=True, normalize=False, binarize=True)
#     H = get_matrix_hermites(X_train, monomials)
#     return ensure_torch(H)

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

def get_y_recon(H, v_tilde, y, classes):
    y_pred = (ensure_torch(H) @ v_tilde).squeeze()
    # err = ((y-y_pred)**2).mean()
    if y.ndim >= 2:
        y = np.argmax(y, axis=1)
    y_non_onehot_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    y_pred_sorted = [y_pred_np[y_non_onehot_np == i] for i in np.unique(y_non_onehot_np)]
    return y_pred_sorted

def sample_v_tilde(top_fra_eigmodes, H, y, n_train, num_trials=20, minmode=True, method="LSTSQ", normalized=True, verbose_every=5, ridge=None):
    Nmax = H.shape[0]
    minval = np.min(top_fra_eigmodes)
    norm_amount = np.sqrt(n_train) if normalized else 1
    
    all_v_tildes = [] if not minmode else torch.zeros(len(top_fra_eigmodes), minval, num_trials)
    for i, top_fra_eigmode in enumerate(top_fra_eigmodes):
        top_fra_eigmode = int(top_fra_eigmode) #usually a np.int which prints exception
        print(f"Starting P={top_fra_eigmode}")
        v_tildes = torch.zeros(top_fra_eigmode, num_trials) if not minmode else torch.zeros(minval, num_trials)
        for trial_idx in range(num_trials):
            if verbose_every is not None and not trial_idx%verbose_every:
                print(f"Starting run {trial_idx}")
            random_sampling = np.random.choice(Nmax, size=n_train, replace=False)
            v_tilde = get_vtilde(H[random_sampling, :top_fra_eigmode], y[random_sampling]/norm_amount, method=method, ridge=ridge)
            v_tildes[:, trial_idx] = v_tilde[:minval] if minmode else v_tilde
        if not minmode:
            all_v_tildes.append(v_tildes) 
        else:
            all_v_tildes[i] = v_tildes
    return all_v_tildes.squeeze()

#plotting stuff

def plot_v_tildes(all_v_tildes, monomials, axes=None, fig=None, titles=None, suptitle="", avg_mode="squared", error_mode="quartiles", colors = None, errorbars=True,
                  **kwargs):
    """
    If axes/fig is None, assumes one plot is wanted; uses plt instead of axes
    """
    def _get_avg_vtilde(v_tildes):
        avg_v = transform_fn(v_tildes).mean(axis=1)
        if error_mode == "quartiles":
            p25 = np.percentile(transform_fn(v_tildes), 25, axis=1)
            p75 = np.percentile(transform_fn(v_tildes), 75, axis=1)
            err_v = np.vstack((ensure_numpy(avg_v) - p25, p75 - ensure_numpy(avg_v))).T
        else:
            err_v = transform_fn(v_tildes).std(axis=1)
        
        num_terms = len(avg_v)
        indices = np.linspace(1, num_terms+1, num_terms)
        degrees = np.array([monomial.degree() for monomial in monomials[:num_terms]])
        return avg_v, err_v, indices, degrees
    
    assert avg_mode in ["squared", "abs"], "Averaging method not found"
    assert error_mode in ["quartiles", "std"], "Error method not found"
    print("Found more v_tildes than axes") if axes is not None and len(all_v_tildes) != len(axes.flatten()) else None
    
    colors = colors if colors is not None else ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]
    titles = titles if titles is not None else len(axes.flatten())*[""]
    
    transform_fn = lambda x: np.abs(x) if avg_mode == "abs" else x**2
    
    ylabhelperavg = "AbsVal of" if avg_mode == "abs" else "Squared"
    ylabhelpershowing = "$|\\tilde{{v}}_i|$" if avg_mode == "abs" else "$\\tilde{{v}}_i^2$"
    ylabhelpervar = "" if errorbars == False else "w/ Squared Error" if error_mode == "squared" else "w/ StdDev Error"
    
    show_values = kwargs.get("show_values", None)

    if axes is not None:
        for i, ax in enumerate(axes.flatten()):
            text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
        
            v_tildes = all_v_tildes[i]
            avg_v, err_v, indices, degrees = _get_avg_vtilde(v_tildes)
            for degree in np.unique(degrees):
                idxs = np.where(np.array(degrees) == degree)[0]
                if errorbars:
                    ax.errorbar(indices[idxs], avg_v[idxs], yerr=np.abs(err_v[idxs].T), color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
                else:
                    ax.scatter(indices[idxs], avg_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
            v_tilde_sum = transform_fn(v_tildes).mean(axis=-1).sum(axis=-1)
            if show_values is not None and np.any([value < len(avg_v) for value in show_values]):
                show_values = np.array([value for value in show_values if value < len(avg_v)])
                if errorbars:
                    ax.errorbar(indices[show_values], avg_v[show_values], yerr=np.abs(err_v[show_values].T), color="xkcd:black", linestyle='', marker='.', alpha=1,)
                else:
                    ax.scatter(indices[show_values], avg_v[show_values], color="xkcd:black", linestyle='', marker='.', alpha=1,)
                formatted_vtilde_vals = " ".join(f"{transform_fn(v_tildes).mean(axis=-1)[show_value].cpu().item()/v_tilde_sum:.2e}" for show_value in show_values)
                ax.text(0.05, 0.825, f'$\\tilde{{v}}_{{{show_values}}}^2/\\Sigma\\tilde{{v}}^2 =$'+formatted_vtilde_vals, **text_kwargs)

            ax.set_title(titles[i])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.text(0.05, 0.9, f'$\\Sigma \\tilde{{v}}^2 = {v_tilde_sum:.2e}$', **text_kwargs)

        fig.suptitle(suptitle, fontsize=24)
        fig.supxlabel("Index", fontsize=18)
        fig.supylabel(f"Average of {ylabhelperavg} Eigencoeff {ylabhelpershowing} {ylabhelpervar}", fontsize=18)
    
    else:
        avg_v, err_v, indices, degrees = _get_avg_vtilde(all_v_tildes)
        for degree in np.unique(degrees):
            idxs = np.where(np.array(degrees) == degree)[0]
            if errorbars:
                plt.errorbar(indices[idxs], avg_v[idxs], yerr=np.abs(err_v[idxs].T), color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
            else:
                plt.scatter(indices[idxs], avg_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
            
        plt.title(titles)
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Index")
        plt.ylabel(f"Average of {ylabhelperavg} Eigencoeff {ylabhelpershowing} {ylabhelpervar}")
    plt.show()

def plot_v_tilde_variances_1(all_v_tildes, degrees, colors=None):
    num_terms = all_v_tildes.shape[1]
    indices = np.linspace(1, num_terms+1, num_terms)
    for i in range(len(all_v_tildes)):
        v_tilde = all_v_tildes[i]
        std_v = (v_tilde**2).std(axis=1)
        for degree in np.unique(degrees):
            idxs = np.where(np.array(degrees) == degree)[0]
            plt.scatter(indices[idxs], std_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=(i+1)/len(all_v_tildes),)
        
    plt.xlabel("Indices")
    plt.ylabel(f"Variance of Eigencoeff Squared $\\tilde{{v}}_i^2$")
    plt.yscale("log")
    plt.title("Opacity ~ N/P (Opaque = Low N/P)")
    plt.show()

def plot_v_tilde_variances_2(all_v_tildes, degrees, PoverNfracs=None, colors=None):
    all_std_v = (all_v_tildes**2).std(axis=-1)
    for degree in np.unique(degrees):
        idxs = np.where(np.array(degrees) == degree)[0]
        plt.plot(np.pow(PoverNfracs, -1), all_std_v[:, idxs].mean(axis=1), color=colors[degree%7], alpha=1, label=f"Degree {degree}")#(degree+1)/6,)
        if all_std_v[:, idxs].squeeze().ndim != 1:
            plt.fill_between(np.pow(PoverNfracs, -1), all_std_v[:, idxs].mean(axis=1)-all_std_v[:, idxs].std(axis=1), all_std_v[:, idxs].mean(axis=1)+all_std_v[:, idxs].std(axis=1),
                                color=colors[degree%7], alpha=((degree+1)/np.max(degrees))*0.4)
        
    plt.xlabel("N/P")
    plt.ylabel(f"Variance of Eigencoeff Squared $\\tilde{{v}}_i^2$")
    plt.xscale("log")
    plt.yscale("log")
    # plt.gca().invert_xaxis()
    plt.legend()
    plt.title(f"Variance of terms vs N/P; Rightwards = further underparameterized")
    plt.show()

#depricated based off changing eigenvectors
# def get_all_vtildes(num_trials, n_trains, top_fra_eigmodes, rng, classes, kerneltype, kernel_width=2):
#     all_v_tildes = []
#     for n in n_trains:
#         print(f"Starting N={n}")
#         for p in top_fra_eigmodes:
#             # indices = np.linspace(1, top_fra_eigmode+1, top_fra_eigmode)
        
#             print(f"Starting P={p}")
#             v_tildes = torch.zeros(p, num_trials)
#             for trial_idx in range(num_trials):
#                 print(f"Starting run {trial_idx}")
#                 if trial_idx == 0:
#                     X_train, y = get_train_dataset(n, dataset_name="cifar10", rng=rng, classes=classes, center=True, normalize=False, binarize=True)
#                     monomials, _, H, _, _ = get_standard_tools(X_train, kerneltype, kernel_width, top_mode_idx=p)
#                     v_tildes[:, trial_idx] = get_vtilde(H, y, method = "LSTSQ", ridge = None)
#                 else:
#                     X_train, y = get_train_dataset(n, dataset_name="cifar10", rng=rng, classes=classes, center=True, normalize=False, binarize=True)
#                     H = ensure_torch(get_matrix_hermites(X_train, monomials))
#                     v_tildes[:, trial_idx] = get_vtilde(H, y, method = "LSTSQ", ridge = None)
                    
#             all_v_tildes.append(v_tildes)