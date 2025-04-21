# from analysis import full_analysis
import matplotlib.pyplot as plt
import numpy as np

#TODO: fix kernelname later
def plot_full_analysis(full_analysis_dict, colors=['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"],
                        axes=None, **plot_kwargs):
    degrees = [monomial.degree() for monomial in full_analysis_dict["Normal"]["monomials"]]
    
    if axes is None:
        _, axes = plt.subplots(2, 2, figsize=(12, 8))

    # eigvals_cpu = eigvals.cpu().numpy()
    # eigvals_g_cpu = eigvals_g.cpu().numpy()
    # eigvals_i_cpu = eigvals_i.cpu().numpy()
    # eigvals_gi_cpu = eigvals_gi.cpu().numpy()
    quartiles = full_analysis_dict["Normal"]["quartiles"]
    quartiles_g = full_analysis_dict["Gaussian"]["quartiles"][:, 1]
    quartiles_i = full_analysis_dict["Independent"]["quartiles"][:, 1]
    quartiles_gi = full_analysis_dict["Gaussian Independent"]["quartiles"][:, 1]
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
    ax.text(0.05, 0.75, f'$\sigma={full_analysis_dict["Normal"]["kernel"].kernel_width:.2f}$', **text_kwargs)
    ax.text(0.05, 0.675, f'$d_\mathrm{{eff}}={d_eff:.2f}$', **text_kwargs)
    axes[0, 1].set_title(f'Gaussianized')
    axes[1, 0].set_title(f'Independentized')
    axes[1, 1].set_title(f'Guassianized, Independentized')
    plt.show()