import numpy as np
import torch as torch
from torchvision import datasets, transforms
from torch.nn.functional import avg_pool2d
from tqdm import tqdm

from .utils.general import ensure_numpy, ensure_torch

def sample_gaussian_data(n_samples, cov_eigvals, target_coeffs=None, noise_std=0):

    Phi = np.random.randn(n_samples, len(cov_eigvals))
    X = Phi * cov_eigvals ** .5

    Y = None

    if target_coeffs is not None:
        Y = Phi @ target_coeffs

        if noise_std > 0:
          Y += noise_std * np.random.randn(*Y.shape)

        if len(Y.shape) == 1:
          Y = Y.reshape(-1, 1)

    return X, Y
