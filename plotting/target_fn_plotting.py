import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import ensure_numpy, ensure_torch

#helper function
def _get_avg_vtilde(v_tildes, mons, error_mode=None, transform_fn=lambda x: x**2):
    avg_v = transform_fn(v_tildes).mean(axis=1)
    if error_mode == "var":
        err_v = v_tildes.var(axis=-1)
    elif error_mode == "quartiles":
        p25 = np.percentile(transform_fn(v_tildes), 25, axis=1)
        p75 = np.percentile(transform_fn(v_tildes), 75, axis=1)
        err_v = np.vstack((ensure_numpy(avg_v) - p25, p75 - ensure_numpy(avg_v))).T
    elif error_mode == "std":
        err_v = transform_fn(v_tildes.std(axis=1))
    else:
        err_v = None
    
    num_terms = len(avg_v)
    indices = np.linspace(1, num_terms+1, num_terms)
    degrees = np.array([monomial.degree() for monomial in mons[:num_terms]])
    return avg_v, err_v, indices, degrees

def plot_eigencoeffs(eigcoeff_dict, title="", avg_mode="squared",
                    colors = None, noise_floor=False, **kwargs):
    """
    Intended to be used alongside get_eigencoeffs() to plot the eigencoefficients against the noise floor.
    """
    assert avg_mode in ["squared", "abs"], "Averaging method not found"
    colors = colors if colors is not None else ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]
    
    transform_fn = lambda x: np.abs(x) if avg_mode == "abs" else x**2
    avg_v, _, indices, degrees = _get_avg_vtilde(eigcoeff_dict['eigencoeffs'], eigcoeff_dict['monomials'])

    for degree in np.unique(degrees):
        idxs = np.where(np.array(degrees) == degree)[0]
        plt.scatter(indices[idxs], avg_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
    
    plt.plot(np.linspace(1, len(avg_v)+1, len(avg_v)), eigcoeff_dict['eigcoeff_var']*np.ones(len(avg_v)), color='xkcd:slate', alpha=1, zorder=4)

    plt.title(title)
    plt.xlabel("Index")
    ylabel = "Eigencoeff Squared $\\hat{{v}}_i^2$" if avg_mode == "squared" else "AbsVal of Eigencoeff $|\\hat{{v}}_i|$"
    plt.ylabel(ylabel)
    plt.xscale("log")
    plt.yscale("log")    

#Note: this is refactored, there used to be functionality when axes was not defined.
#      if axes is not given, it will no longer use plt.plot()
def plot_multiple_eigencoeffs(all_v_tildes, monomials, axes=None, fig=None, titles=None, suptitle="", avg_mode="squared",
                     error_mode="var", colors = None, errorbars=True, **kwargs):
    """
    Plots eigencoefficients across multiple runs.
    """
    
    assert avg_mode in ["squared", "abs"], "Averaging method not found"
    assert error_mode in ["quartiles", "std", "var"], "Error method not found"
    print("Found more v_tildes than axes") if axes is not None and len(all_v_tildes) != len(axes.flatten()) else None
    
    ## defaulting colors, titles if none given
    colors = colors if colors is not None else ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]
    titles = titles if titles is not None else len(axes.flatten())*[""]    
    
    transform_fn = lambda x: np.abs(x) if avg_mode == "abs" else x**2
    
    ## useful such that the ylabel doesn't have to be manually defined
    ylabhelperavg = "AbsVal of" if avg_mode == "abs" else "Squared"
    ylabhelpershowing = "$|\\tilde{{v}}_i|$" if avg_mode == "abs" else "$\\tilde{{v}}_i^2$"
    ylabhelpervar = "" # case for when there are no errorbars
    if errorbars:
        ylabhelpervar = "w/ Variance Error" if error_mode == "var" else "w/ StdDev Error" if error_mode == "std" else "w/ Quartile Error"
    
    ## if we have multiple different datasets we want to plot, the monomials will appear slightly differently
    ## to get around this, we'll have multiple monomials able to be defined; defaulted to just monomials otherwise
    monlist = isinstance(monomials, list) and all(isinstance(elem, list) for elem in monomials)
    mons = monomials

    for i, ax in enumerate(axes.flatten()):
        text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
        if monlist:
            mons = monomials[i%len(monomials)]
    
        v_tildes = all_v_tildes[i]
        avg_v, err_v, indices, degrees = _get_avg_vtilde(v_tildes, mons, error_mode, transform_fn)
        for degree in np.unique(degrees):
            idxs = np.where(np.array(degrees) == degree)[0]
            if errorbars:
                ax.errorbar(indices[idxs], avg_v[idxs], yerr=np.abs(err_v[idxs].T), color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
            else:
                ax.scatter(indices[idxs], avg_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)

        ax.set_title(titles[i])
        ax.set_xscale("log")
        ax.set_yscale("log")
        v_tilde_sum = transform_fn(v_tildes).mean(axis=-1).sum(axis=-1)
        ax.text(0.7, 0.9, f'$\\Sigma \\tilde{{v}}^2 = {v_tilde_sum:.2e}$', **text_kwargs)

    fig.suptitle(suptitle, fontsize=24)
    fig.supxlabel("Index", fontsize=18)
    fig.supylabel(f"Average of {ylabhelperavg} Eigencoeff {ylabhelpershowing} {ylabhelpervar}", fontsize=18)
    
def plot_true_eigencoeffs(v_true, axes, clipper):
    """
    Plots true eigencoefficients in a solid black line; forces to foreground.
    """
    for i, ax in enumerate(axes):
        text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
        clipper = clipper[i]
        v_true_clipped = (v_true**2)[:clipper].cpu().numpy()
        ax.plot(np.linspace(1, clipper+1, clipper), v_true_clipped, color='xkcd:black', zorder=10)
        ax.text(0.7, 0.8, f'$\\Sigma \\tilde{{v}}_{{True}}^2 = {v_true_clipped.sum():.2e}$', **text_kwargs)

def show_value_on_eigencoeff_plot(index, v_tilde, axes, indices, errorbars=True, num_shown=0):
    #possibility that axes are of different xlengths, so we need to check that the index is within bounds
    if index > len(indices):
        return None
    avg_v = (v_tilde**2).mean(axis=1)
    err_v = v_tilde.var(axis=-1)
    for i, ax in enumerate(axes.flatten()):
        text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
        if np.any([value < len(avg_v) for value in index]):
            if errorbars:
                ax.errorbar(indices[index], avg_v, yerr=np.abs(err_v), color="xkcd:black", linestyle='', marker='.', alpha=1,)
            else:
                ax.scatter(indices[index], avg_v, color="xkcd:black", linestyle='', marker='.', alpha=1,)
            ax.text(0.05, 0.9-.075*num_shown, f'$\\tilde{{v}}_{{{index}}}^2 = {avg_v.cpu().item():.2e}$', **text_kwargs)

def plot_eigencoeff_variance_opacity(all_v_tildes, degrees, colors=None):
    """
    Uses an opacity plot to see how the eigencoeff variation changes over the first axis.
    """
    num_terms = all_v_tildes.shape[1]
    indices = np.linspace(1, num_terms+1, num_terms)
    for i in range(len(all_v_tildes)):
        v_tilde = all_v_tildes[i]
        std_v = (v_tilde).var(axis=1)
        for degree in np.unique(degrees):
            idxs = np.where(np.array(degrees) == degree)[0]
            plt.scatter(indices[idxs], std_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=(i+1)/len(all_v_tildes),)
        
    plt.xlabel("Indices")
    plt.ylabel(f"Variance of Eigencoeff $\\tilde{{v}}_i$")
    plt.yscale("log")
    plt.title("Opacity ~ P/N (Highly Opaque = High Pbar)")

def plot_eigencoeff_variance_level(all_v_tildes, degrees, Pbars=None, colors=None):
    """
    Grabs all eigencoefficients given of the same level and plots against Pbar
    """
    all_var_v = (all_v_tildes).var(axis=-1)
    for degree in np.unique(degrees):
        idxs = np.where(np.array(degrees) == degree)[0]
        plt.plot(Pbars, all_var_v[:, idxs].mean(axis=1), color=colors[degree%7], alpha=1, label=f"Degree {degree}")#(degree+1)/6,)
        if all_var_v[:, idxs].squeeze().ndim != 1:
            plt.fill_between(Pbars, all_var_v[:, idxs].mean(axis=1)-all_var_v[:, idxs].std(axis=1), all_var_v[:, idxs].mean(axis=1)+all_var_v[:, idxs].std(axis=1),
                                color=colors[degree%7], alpha=((degree+1)/np.max(degrees))*0.4)
        
    plt.xlabel("Pbar")
    plt.ylabel(f"Variance of Eigencoeff Squared $\\tilde{{v}}_i^2$")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title(f"Variance of terms vs N/P; Rightwards = further underparameterized")