from itertools import product
import numpy as np
import torch

from utils.general import ensure_torch
from utils.hermite import generate_fra_monomials, get_matrix_hermites

def generate_configs(params):
    # Define the order of keys
    key_order = ["d", "N", "offset", "alpha", "bandwidth", "kerneltype"]

    # Separate static and dynamic parameters
    dynamics = {k: v for k, v in params.items() if isinstance(v, (list, tuple, np.ndarray))}
    statics = {k: v for k, v in params.items() if not isinstance(v, (list, tuple))}

    # Generate all combinations of dynamic values
    dynamic_keys = list(dynamics.keys())
    dynamic_combinations = product(*dynamics.values())

    # Construct configurations
    config_list = []
    for combo in dynamic_combinations:
        config = dict(zip(dynamic_keys, combo))
        config.update(statics)  # Add static values
        config_list.append(tuple(config[k] for k in key_order))

    # Create a nested list structure for each dynamic key
    def nest_configs(configs, keys):
        if not keys:
            return configs
        grouped = {}
        key_idx = key_order.index(keys[0])
        for cfg in configs:
            grouped.setdefault(cfg[key_idx], []).append(cfg)
        return [nest_configs(v, keys[1:]) for v in grouped.values()]

    return nest_configs(config_list, dynamic_keys)

def generate_structure(config, top_mode_idx = 3000):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, N, offset, alpha, bandwidth, kerneltype = config
    data_eigvals = ensure_torch((offset+np.arange(d)) ** -alpha)
    data_eigvals /= data_eigvals.sum()
    X = ensure_torch(torch.normal(0, 1, (N, d))).to(DEVICE) * torch.sqrt(data_eigvals).to(DEVICE)

    torch.cuda.empty_cache()
    kernel = kerneltype(X, bandwidth=bandwidth)
    eval_level_coeff = kerneltype.get_level_coeff_fn(bandwidth=bandwidth, data_eigvals=data_eigvals)

    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, top_mode_idx, eval_level_coeff)
    H = get_matrix_hermites(X, monomials).to(DEVICE)

    kernel_eigvals, _ = kernel.eigendecomp()

    _, _, quartiles = kernel.kernel_function_projection(H)

    return monomials, kernel_eigvals, quartiles, fra_eigvals, data_eigvals

def plot_structure(config, monomials, kernel_eigvals, quartiles, fra_eigvals, data_eigvals, ax, xlim=None, title=None):
    d, N, offset, alpha, bandwidth, kerneltype = config
    degrees = [monomial.degree() for monomial in monomials]
    colors = ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]

    eigvals_cpu = kernel_eigvals.cpu().numpy()
    eff_eigvals = quartiles[:, 1]
    xerr = -np.array([quartiles[:, 0]-eff_eigvals, eff_eigvals-quartiles[:, 2]])
    for degree in np.unique(degrees):
        idxs = np.where(np.array(degrees) == degree)[0]
        ax.errorbar(eff_eigvals[idxs], fra_eigvals[idxs], xerr=xerr[:, idxs],
                    color=colors[degree%7], linestyle='', marker='o')

    eigvals_xspace = kernel_eigvals[:len(fra_eigvals)].cpu().numpy() #len(fra) == top_mode_idx
    ax.plot(eigvals_xspace, eigvals_xspace, color='xkcd:slate', alpha=0.3, zorder=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    if xlim is not None: ax.set_xlim(xlim[0], xlim[1])
    d_eff = (data_eigvals.sum())**2/(data_eigvals**2).sum()
    if title is not None: ax.set_title(title)
    text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
    ax.text(0.05, 0.9, f'$d={d}$', **text_kwargs)
    ax.text(0.05, 0.825, f'$N={N}$', **text_kwargs)
    ax.text(0.05, 0.75, f'$\\alpha={alpha}$', **text_kwargs)
    ax.text(0.05, 0.675, f'$\sigma={bandwidth:.2f}$', **text_kwargs)
    ax.text(0.05, 0.6, f'$i_0={offset}$', **text_kwargs)
    ax.text(0.05, 0.525, f'$d_\mathrm{{eff}}={d_eff:.2f}$', **text_kwargs)