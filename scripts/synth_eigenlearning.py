import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange, reduce, repeat

import sys
import os
from tqdm import tqdm

sys.path.append("../")

from ImageData import ImageData, preprocess
from ExptTrace import ExptTrace
from kernels import GaussianKernel, LaplaceKernel, ExponentialKernel, krr
from feature_decomp import generate_fra_monomials, Monomial
from utils import ensure_torch, get_matrix_hermites
from eigenlearning import eigenlearning
to_torch = ensure_torch


DATA_PATH = os.getenv("DATASETPATH")
EXPT_PATH = os.getenv("EXPTPATH")
if DATA_PATH is None:
    raise ValueError("must set $DATASETPATH environment variable")
if EXPT_PATH is None:
    raise ValueError("must set $EXPTPATH environment variable")
main_dir = os.path.join(os.getenv("EXPTPATH"), "phlab")
expt_name = 'test'
expt_dir = f'{main_dir}/{expt_name}'

for dir in [main_dir, expt_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

def emp_eigsys(kernel, y):
    eigvals, eigvecs = kernel.eigendecomp()
    eigcoeffs = eigvecs.T @ y
    eigcoeffs /= torch.linalg.norm(eigcoeffs)
    eigvals = eigvals.cpu().numpy()
    eigcoeffs = eigcoeffs.cpu().numpy()

    return eigvals, eigcoeffs

def fra_eigsys(X, y, eval_level_coeff):
    N, _ = X.shape
    S = torch.linalg.svdvals(X)
    data_eigvals = S**2 / (S**2).sum()

    eigvals, monomials = generate_fra_monomials(data_eigvals, N, eval_level_coeff, kmax=10)
    H = get_matrix_hermites(X, monomials)
    eigcoeffs = torch.linalg.lstsq(H, y).solution
    eigcoeffs /= torch.linalg.norm(eigcoeffs)
    eigcoeffs = eigcoeffs.cpu().numpy()

    return eigvals, eigcoeffs


def learning_curve(ntrains, eigvals, eigcoeffs, ridge=0, noise_var=0):
    kappas, learnabilities, e0s, train_mses, test_mses = [np.zeros(len(ntrains)) for _ in range(5)]
    for i, n in enumerate(ntrains):
        res = eigenlearning(n, eigvals, eigcoeffs, ridge, noise_var)
        # kappas[i] = res["kappa"]
        # learnabilities[i] = res["learnability"]
        # e0s[i] = res["overfitting_coeff"]
        train_mses[i] = res["train_mse"]
        test_mses[i] = res["test_mse"]
    return train_mses, test_mses

N = 3000 # 12000
d = 200 # 500
offset = 3
alpha = 1.3
kerneltype = GaussianKernel
kwidth = 4

data_eigvals = to_torch((offset+np.arange(d)) ** -alpha)
data_eigvals /= data_eigvals.sum()
X = to_torch(torch.normal(0, 1, (N, d))) * torch.sqrt(data_eigvals)
S = torch.linalg.svdvals(X)
# to make norm(x)~1 on average (level_coeff eqn requires this)
X = X * torch.sqrt(N / (S**2).sum())

kernel = kerneltype(X, kernel_width=kwidth)

P = N
eval_level_coeff = kerneltype.get_level_coeff_fn(data_eigvals, kernel_width=kwidth)
eigvals, monomials = generate_fra_monomials(data_eigvals, P, eval_level_coeff, kmax=10)
H = get_matrix_hermites(X, monomials)
offset = 6
beta = 0.53
synth_eigcoeffs = to_torch((offset+np.arange(P)) ** -beta)
synth_eigcoeffs[0] = 0
synth_eigcoeffs /= torch.linalg.norm(synth_eigcoeffs)
y = np.sqrt(N) * H @ synth_eigcoeffs

noise_var = 2e-1
noise = to_torch(torch.normal(0, np.sqrt(noise_var), y.shape))
SNR = torch.linalg.norm(y) / torch.linalg.norm(noise)
print(f"SNR = {SNR}")
y += noise