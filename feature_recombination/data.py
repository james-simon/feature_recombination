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

def pca_rotate(X):
    """
    Perform PCA rotation on the input matrix X and return PCA eigenvalues.

    Args:
        X (np.ndarray or torch.Tensor): Input matrix.

    Returns:
        X_rotated (np.ndarray or torch.Tensor): Rotated matrix.
        eigenvalues (np.ndarray or torch.Tensor): PCA eigenvalues.
    """
    # Check the input type and ensure tensor
    is_numpy = isinstance(X, np.ndarray)
    X = ensure_torch(X)

    # Perform SVD
    U, S, Vh = torch.linalg.svd(X)

    # Rotate X
    X_rotated = X @ Vh.T

    # Compute PCA eigenvalues
    eigenvalues = S ** 2 / X.shape[0]

    # Convert back to numpy if the input was numpy
    if is_numpy:
        X_rotated = ensure_numpy(X_rotated)
        eigenvalues = ensure_numpy(eigenvalues)

    return X_rotated, eigenvalues
