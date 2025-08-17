# from analysis import full_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from analysis.gaussianity_tests import eigvecs_to_gaussian
from analysis.independence_tests import eigvecs_to_independent

#TODO: fix kernelname later
def plot_full_analysis(full_analysis_dict, colors=['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"],
                        axes=None, **plot_kwargs):
    degrees = [monomial.degree() for monomial in full_analysis_dict["Normal"]["monomials"]]
    
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8))

    quartiles = full_analysis_dict["Normal"]["quartiles"]
    quartiles_g = full_analysis_dict["Gaussian"]["quartiles"]
    quartiles_i = full_analysis_dict["Independent"]["quartiles"]
    quartiles_gi = full_analysis_dict["Gaussian Independent"]["quartiles"]
    eff_eigvals = quartiles[:, 1]
    eff_eigvals_g = quartiles_g[:, 1]
    eff_eigvals_i = quartiles_i[:, 1]
    eff_eigvals_gi = quartiles_gi[:, 1]
    xerr = -np.array([quartiles[:, 0]-eff_eigvals, eff_eigvals-quartiles[:, 2]])
    xerr_g = -np.array([quartiles_g[:, 0]-eff_eigvals_g, eff_eigvals_g-quartiles_g[:, 2]])
    xerr_i = -np.array([quartiles_i[:, 0]-eff_eigvals_i, eff_eigvals_i-quartiles_i[:, 2]])
    xerr_gi = -np.array([quartiles_gi[:, 0]-eff_eigvals_gi, eff_eigvals_gi-quartiles_gi[:, 2]])

    xmin, xmax = 1e-7, 2e0
    xx = np.linspace(xmin, xmax, 10)
    
    def plot_eigvals(ax, eff_eigvals, fra_eigvals, xerr):
        for degree in np.unique(degrees):
            idxs = np.where(np.array(degrees) == degree)[0]
            markers, caps, bars = ax.errorbar(eff_eigvals[idxs], fra_eigvals[idxs], xerr=xerr[:, idxs],
                        color=colors[degree%7], linestyle='', marker='.', alpha=1,)
            [bar.set_alpha(0.2) for bar in bars]
        
        ax.set_xlim(xmin, xmax)
        ax.plot(xx, xx, color='xkcd:slate', alpha=0.4, zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')

    plot_eigvals(axes[0, 0], eff_eigvals, full_analysis_dict["Normal"]["fra_eigvals"], xerr)
    plot_eigvals(axes[0, 1], eff_eigvals_g, full_analysis_dict["Gaussian"]["fra_eigvals"], xerr_g)
    plot_eigvals(axes[1, 0], eff_eigvals_i, full_analysis_dict["Independent"]["fra_eigvals"], xerr_i)
    plot_eigvals(axes[1, 1], eff_eigvals_gi, full_analysis_dict["Gaussian Independent"]["fra_eigvals"], xerr_gi)

    
    d_eff = (full_analysis_dict["Normal"]["data_eigvals"].sum())**2/(full_analysis_dict["Normal"]["data_eigvals"]**2).sum()
    ax = axes[0, 0]
    ax.set_title(f'{plot_kwargs["kernelname"]} @ {plot_kwargs["dataname"]}')
    text_kwargs = {'fontsize': 12, 'transform': ax.transAxes}
    ax.text(0.05, 0.9, f'$d={plot_kwargs["d"]}$', **text_kwargs)
    ax.text(0.05, 0.825, f'$N={plot_kwargs["N"]}$', **text_kwargs)
    ax.text(0.05, 0.75, f'$\sigma={plot_kwargs["kernel_width"]:.2f}$', **text_kwargs)
    ax.text(0.05, 0.675, f'$d_\mathrm{{eff}}={d_eff:.2f}$', **text_kwargs)
    axes[0, 1].set_title(f'Gaussianized')
    axes[1, 0].set_title(f'Independentized')
    axes[1, 1].set_title(f'Guassianized, Independentized')
    plt.show()

def plot_full_pca_distributions(X):
    N, d = X.shape

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    X = np.sqrt(N) * U @ torch.diag(S)
    eigvecs_gaussian = eigvecs_to_gaussian(X, S).cpu()
    eigvecs_independent = eigvecs_to_independent(X, bsz=X.shape[0]).cpu()
    eigvecs_indep_gaussian = eigvecs_to_gaussian(eigvecs_independent, S).cpu()

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 9), sharex=True, sharey=True)

    def plot_vals(arr, i, title=None):
        xvar = S[i].item()**2
        Xi = arr[:, i].cpu().numpy() / np.sqrt(xvar)
        Xi = Xi[np.abs(Xi) <= 3.1]
        ax.hist(Xi, bins=100,
                density=True, histtype='step', cumulative=False)

        ax.set_title(title)
        ax.set_xlim(-3, 3)

    for i, ax in enumerate(axes.flatten()):
        if i//4 == 0:
            xvar = S[i].item()**2
            plot_vals(X, i, f"$\lambda^\mathrm{{(PCA)}}_{{{i}}}={xvar:.4f}$")
        elif i//4 == 1:
            xvar = S[i%4].item()**2
            plot_vals(eigvecs_gaussian, i%4, f"$\lambda^\mathrm{{(Gaussian)}}_{{{i%4}}}={xvar:.4f}$")
        elif i//4 == 2:
            xvar = S[i%4].item()**2
            plot_vals(eigvecs_independent, i%4, f"$\lambda^\mathrm{{(Independent)}}_{{{i%4}}}={xvar:.4f}$")
        elif i//4 == 3:
            xvar = S[i%4].item()**2
            plot_vals(eigvecs_indep_gaussian, i%4, f"$\lambda^\mathrm{{(IndepGauss)}}_{{{i%4}}}={xvar:.4f}$")

    plt.tight_layout()
    plt.show()

    return X, eigvecs_gaussian, eigvecs_independent, eigvecs_indep_gaussian, Vt

def plot_heatmaps(eigvecs, eigvecs_gaussian, eigvecs_independent, eigvecs_indep_gaussian, Vt):
    
    X_recon = eigvecs.cpu() @ Vt.cpu()
    X_gaussian_recon = eigvecs_gaussian @ Vt.cpu().numpy()
    X_indep_recon = eigvecs_independent @ Vt.cpu().numpy()
    X_gaussian_indep_recon = eigvecs_indep_gaussian @ Vt.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.heatmap(X_recon, ax=axes[0, 0], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[0, 0].set_title("X_recon")

    sns.heatmap(X_gaussian_recon, ax=axes[0, 1], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[0, 1].set_title("X_gaussian_recon")

    sns.heatmap(X_indep_recon, ax=axes[1, 0], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[1, 0].set_title("X_indep_recon")

    sns.heatmap(X_gaussian_indep_recon, ax=axes[1, 1], cmap='viridis', xticklabels=False, yticklabels=False)
    axes[1, 1].set_title("X_gaussian_indep_recon")

    plt.tight_layout()
    plt.show()

    return X_recon, X_gaussian_recon, X_indep_recon, X_gaussian_indep_recon