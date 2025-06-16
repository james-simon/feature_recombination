import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import ensure_numpy, ensure_torch

def plot_v_tildes(all_v_tildes, monomials, axes=None, fig=None, titles=None, suptitle="", avg_mode="squared", error_mode="quartiles", colors = None, errorbars=True,
                  v_true=None, **kwargs):
    """
    If axes/fig is None, assumes one plot is wanted; uses plt instead of axes
    """
    def _get_avg_vtilde(v_tildes, mons):
        avg_v = transform_fn(v_tildes).mean(axis=1)
        if error_mode == "quartiles":
            p25 = np.percentile(transform_fn(v_tildes), 25, axis=1)
            p75 = np.percentile(transform_fn(v_tildes), 75, axis=1)
            err_v = np.vstack((ensure_numpy(avg_v) - p25, p75 - ensure_numpy(avg_v))).T
        else:
            err_v = transform_fn(v_tildes).std(axis=1)
        
        num_terms = len(avg_v)
        indices = np.linspace(1, num_terms+1, num_terms)
        degrees = np.array([monomial.degree() for monomial in mons[:num_terms]])
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

    monlist = isinstance(monomials, list) and all(isinstance(elem, list) for elem in monomials)
    mons = monomials

    if axes is not None:
        for i, ax in enumerate(axes.flatten()):
            text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
            if monlist:
                mons = monomials[i//len(monomials)]
        
            v_tildes = all_v_tildes[i]
            avg_v, err_v, indices, degrees = _get_avg_vtilde(v_tildes, mons)
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
            ax.text(0.7, 0.9, f'$\\Sigma \\tilde{{v}}^2 = {v_tilde_sum:.2e}$', **text_kwargs)
            if v_true is not None:
                clipper = kwargs.get("clipper", None)
                clipper = clipper[i] if clipper is not None else v_tildes.shape[0]
                v_true_clipped = transform_fn(v_true)[:clipper].cpu().numpy()
                ax.plot(np.linspace(1, clipper+1, clipper), v_true_clipped, color='xkcd:black', zorder=10)
                ax.text(0.7, 0.8, f'$\\Sigma \\tilde{{v_{{True}}}}^2 = {v_true_clipped.sum():.2e}$', **text_kwargs)


        fig.suptitle(suptitle, fontsize=24)
        fig.supxlabel("Index", fontsize=18)
        fig.supylabel(f"Average of {ylabhelperavg} Eigencoeff {ylabhelpershowing} {ylabhelpervar}", fontsize=18)
    
    else:
        avg_v, err_v, indices, degrees = _get_avg_vtilde(all_v_tildes, mons)
        for degree in np.unique(degrees):
            idxs = np.where(np.array(degrees) == degree)[0]
            if errorbars:
                plt.errorbar(indices[idxs], avg_v[idxs], yerr=np.abs(err_v[idxs].T), color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
            else:
                plt.scatter(indices[idxs], avg_v[idxs], color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
            
        if v_true is not None:
            clipper = kwargs.get("clipper", None)
            clipper = clipper[i] if clipper is not None else v_tildes.shape[0]
            v_true_clipped = transform_fn(v_true)[:clipper].cpu().numpy()
            plt.plot(np.linspace(1, clipper+1, clipper), v_true_clipped, color='xkcd:black', zorder=10)
            plt.text(0.7, 0.8, f'$\\Sigma \\tilde{{v_{{True}}}}^2 = {v_true_clipped.sum():.2e}$', **{'fontsize': 12})
            
        plt.title(titles)
        plt.xscale("log")
        plt.yscale("log")

        plt.xlabel("Index")
        plt.ylabel(f"Average of {ylabhelperavg} Eigencoeff {ylabhelpershowing} {ylabhelpervar}")
    

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