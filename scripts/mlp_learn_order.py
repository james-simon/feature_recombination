import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

import sys

sys.path.append("../")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rc("figure", dpi=100, facecolor=(1, 1, 1))
plt.rc("font", family='stixgeneral', size=14)
plt.rc("axes", facecolor=(1, .99, .95), titlesize=18)
plt.rc("mathtext", fontset='cm')
from kernels import GaussianKernel
from data import get_synthetic_dataset, get_synthetic_X, get_matrix_hermites
from feature_decomp import Monomial, generate_fra_monomials
from utils import ensure_torch, ensure_numpy

from ExptTrace import ExptTrace
from mupify import mupify, rescale

dim = 10
N = 10_000
offset=3
alpha=1.2
bsz = 512

X, y, locs, H, monomials, fra_eigvals, data_eigvals = get_all_targets(target_monomials=Monomial({}), dim=dim, N=N)
target_monomials = monomials[:10]
lr = 1e-2
width = 8192
gamma = 1

colors = ['xkcd:red', 'xkcd:orange', 'xkcd:gold', 'xkcd:green', 'xkcd:blue', "xkcd:purple", "xkcd:black"]

num_trials = 2
max_iter = int(5e4)
loss_checkpoints = (5e-1, 2e-2)
percent_thresholds = (0.75, 0.5)

X, y, locs, H, monomials, fra_eigvals, data_eigvals = get_all_targets(target_monomials=target_monomials, dim=dim, N=N)
U, lambdas, Vt = torch.linalg.svd(X, full_matrices=False)
degrees = np.array([monomial.degree() for monomial in monomials[:int(locs[-1])]])
indices = np.linspace(1, int(locs[-1])+1, int(locs[-1]))
for degree in np.unique(degrees):
    idxs = np.where(np.array(degrees) == degree)[0]
    plt.scatter(indices[idxs], fra_eigvals[idxs].cpu(), color=colors[degree%7], linestyle='', marker='.', alpha=1/(degree+1),)
plt.yscale("log")
plt.xscale('log')
all_losses = np.ones((len(target_monomials), num_trials, max_iter))*max(loss_checkpoints)
pct_breakpoints = np.empty((len(target_monomials), len(percent_thresholds), num_trials))
pbar = tqdm(enumerate(target_monomials), total=len(target_monomials), desc="Processing items")

for idx, target_monomial in pbar:
    for trial in range(num_trials):
        model = MLP(d_in=dim, depth=2, d_out=1, width=width).to(device)
        _, losses, pct_breakpoint = train_network(model=model, lambdas=lambdas, Vt=Vt, monomial=target_monomial, dim=dim, bsz=bsz,
                                                  data_eigvals=data_eigvals, N=N, lr=lr, max_iter=max_iter,
                                                  percent_thresholds=percent_thresholds, gamma=gamma, ema_smoother=0.9)
        all_losses[idx, trial] = losses
        pct_breakpoints[idx, :, trial] = pct_breakpoint
    pbar.set_postfix(current_item=target_monomial, iteration=idx + 1)
    if not(idx%10) and idx!=0:
        fig, axes = plot_time_vs_error(all_losses, target_monomials, fra_eigvals, locs, plotindex=idx)
        fig.suptitle(f"Online μP MLP training on synth monomials averaged over {num_trials} runs, $\\gamma$ = {gamma}")
        plt.tight_layout(); plt.show()
        slope, intercept, xaxis = get_slope_and_intercept(fra_eigvals, locs, pct_breakpoints, breakindex = 1, trainindex = idx)
        plot_eigval_vs_traintime(slope, intercept, xaxis, fra_eigvals, locs, pct_breakpoints, target_monomials, breakindex = 1, trainindex = idx)
        plt.show()
        print(f"Breakpoints: {pct_breakpoints[:idx, 1, :].mean(axis=-1)}")
fig, axes = plot_time_vs_error(all_losses, target_monomials, fra_eigvals, locs, plotindex=idx)
fig.suptitle(f"Online μP MLP training on synth monomials averaged over {num_trials} runs, $\\gamma$ = {gamma}")
plt.tight_layout(); plt.show()
slope, intercept, xaxis = get_slope_and_intercept(fra_eigvals, locs, pct_breakpoints, breakindex = 1, trainindex = idx)
plot_eigval_vs_traintime(slope, intercept, xaxis, fra_eigvals, locs, pct_breakpoints, target_monomials, breakindex = 1, trainindex = idx)
plt.show()
print(f"Breakpoints: {pct_breakpoints[:idx, 1, :].mean(axis=-1)}")