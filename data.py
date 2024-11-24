import numpy as np
import torch as torch
from torchvision import datasets, transforms
from torch.nn.functional import avg_pool2d
from tqdm import tqdm

from .utils import ensure_numpy, ensure_torch

def sample_gaussian_data(n_samples, cov_eigvals, target_coeffs, noise_std=0):

    Phi = np.random.randn(n_samples, len(cov_eigvals))
    X = Phi * cov_eigvals ** .5
    Y = Phi @ target_coeffs

    if noise_std > 0:
      Y += noise_std * np.random.randn(*Y.shape)

    if len(Y.shape) == 1:
      Y = Y.reshape(-1, 1)

    return X, Y

def load_image_dataset(n_samples, dataset_name, downsampling_factor=1):
    """
    Load n_samples images from MNIST or CIFAR10 datasets as flattened NumPy arrays.
    Optionally downsample the images by the specified factor.

    Parameters:
    - n_samples (int): Number of images to load.
    - dataset_name (str): Name of the dataset ('MNIST' or 'CIFAR10').
    - downsampling_factor (int): Factor by which to downsample the images.

    Returns:
    - images (np.ndarray): Flattened images of shape (n_samples, downsampled_image_size).
    - labels (np.ndarray): Corresponding labels of the images.
    """
    if dataset_name not in ['MNIST', 'CIFAR10']:
        raise ValueError("dataset_name must be 'MNIST' or 'CIFAR10'")

    def downsample(img_tensor):
        """
        Downsample the image tensor by the given factor using average pooling.
        """
        if downsampling_factor > 1:
            # Add batch dimension, apply avg_pool2d, and flatten
            pooled = avg_pool2d(img_tensor.unsqueeze(0), kernel_size=downsampling_factor, stride=downsampling_factor)
            return pooled.squeeze(0).flatten().numpy()
        return img_tensor.flatten().numpy()

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Lambda(lambda x: downsample(x))
    ])

    # Load dataset
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Ensure n_samples is within dataset size
    if n_samples > len(dataset):
        raise ValueError(f"n_samples must be <= total samples in {dataset_name} ({len(dataset)})")

    # Load and stack samples
    images, labels = zip(*[dataset[i] for i in range(n_samples)])
    return np.stack(images), np.array(labels)

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
    eigenvalues = S ** 2

    # Convert back to numpy if the input was numpy
    if is_numpy:
        X_rotated = ensure_numpy(X_rotated)
        eigenvalues = ensure_numpy(eigenvalues)

    return X_rotated, eigenvalues