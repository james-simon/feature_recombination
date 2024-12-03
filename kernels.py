import numpy as np
import torch
from tqdm import tqdm

from .utils import ensure_numpy, ensure_torch

def exponential_kernel(X1, X2, width):
    X1, X2 = ensure_numpy(X1), ensure_numpy(X2)
    K_lin = X1 @ X2.T
    return np.exp(K_lin / width ** 2)

def gaussian_kernel(X1, X2, width, chunk_size=1000):
    X1, X2 = ensure_numpy(X1), ensure_numpy(X2)
    n_samples1, n_samples2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n_samples1, n_samples2), dtype=np.float32)

    # Compute the kernel matrix in chunks
    for i in tqdm(range(0, n_samples1, chunk_size)):
        for j in range(0, n_samples2, chunk_size):
            X1_chunk = X1[i:i + chunk_size]
            X2_chunk = X2[j:j + chunk_size]
            diffs = X1_chunk[:, None, :] - X2_chunk[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            K[i:i + chunk_size, j:j + chunk_size] = np.exp(-0.5 * (dists / width) ** 2)

    return K

def laplace_kernel(X1, X2, width, chunk_size=1000):
    X1, X2 = ensure_numpy(X1), ensure_numpy(X2)
    n_samples1, n_samples2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n_samples1, n_samples2), dtype=np.float32)

    # Compute the kernel matrix in chunks
    for i in tqdm(range(0, n_samples1, chunk_size)):
        for j in range(0, n_samples2, chunk_size):
            X1_chunk = X1[i:i + chunk_size]
            X2_chunk = X2[j:j + chunk_size]
            diffs = X1_chunk[:, None, :] - X2_chunk[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            K[i:i + chunk_size, j:j + chunk_size] = np.exp(-dists / width)

    return K

def linear_kernel(X1, X2):
    X1, X2 = ensure_numpy(X1), ensure_numpy(X2)
    return X1 @ X2.T

def krr(K, y, n_train, ridge=0, dtype=torch.float64, debug=False):
    """
    Kernel Ridge Regression (KRR) with support for different data types.

    Args:
        K (array-like or torch.Tensor): Kernel matrix.
        y (array-like or torch.Tensor): Target values.
        n_train (int): Number of training samples.
        ridge (float): Regularization parameter. Default is 0 (no regularization).
        dtype (torch.dtype): Data type, e.g., torch.float32 or torch.float64. Default is torch.float32.
        debug (bool): If True, enters debug mode with pdb.

    Returns:
        tuple: train_mse, test_mse, test_lrn
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors if not already, and ensure correct dtype and device
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=dtype).to(DEVICE)
    else:
        K = K.to(dtype=dtype, device=DEVICE)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=dtype).to(DEVICE)
    else:
        y = y.to(dtype=dtype, device=DEVICE)

    K_train, K_test = K[:n_train, :n_train], K[:, :n_train]
    y_train, y_test = y[:n_train], y[n_train:]

    if ridge == 0:
        alpha = torch.linalg.lstsq(K_train, y_train)
    else:
        eye = torch.eye(n_train, dtype=dtype).to(DEVICE)
        alpha = torch.linalg.lstsq(K_train + ridge * eye, y_train)

    y_hat = K_test @ alpha.solution
    # Train error
    y_hat_train = y_hat[:n_train]
    train_mse = ((y_train - y_hat_train) ** 2).mean(axis=0).cpu().numpy()

    # Test error
    y_hat_test = y_hat[n_train:]
    test_mse = ((y_test - y_hat_test) ** 2).mean(axis=0).cpu().numpy()

    test_lrn = (y_test * y_hat_test).mean(axis=0) / (y_test ** 2).mean(axis=0)
    test_lrn = test_lrn.cpu().numpy()

    if debug:
        import pdb;
        pdb.set_trace()

    return train_mse, test_mse, test_lrn

def compute_learning_curve(ns, n_test, K, Y, ridge=0):
  train_mses, test_mses, test_lrns = [], [], []

  for n_train in tqdm(ns):
    train_mse, test_mse, test_lrn = krr(K[:n_train+n_test,:n_train+n_test], Y[:n_train+n_test], n_train, ridge=ridge)

    train_mses.append(train_mse)
    test_mses.append(test_mse)
    test_lrns.append(test_lrn)

  train_mses = np.array(train_mses).T
  test_mses = np.array(test_mses).T
  test_lrns = np.array(test_lrns).T

  return {
      'train_mse': train_mses,
      'test_mse': test_mses,
      'test_lrn': test_lrns
  }

def kappa_trace(K, ns, ridge, dtype=torch.float64):
    """
    Compute the experimental kappa values for a range of sizes with ridge regularization.

    Args:
        K (array-like or torch.Tensor): Kernel matrix.
        ns (list of int): List of sizes to compute kappa for.
        ridge (float): Regularization parameter.
        dtype (torch.dtype): Data type, e.g., torch.float32 or torch.float64.

    Returns:
        np.ndarray: Array of kappa values for each size in ns.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert K to a torch tensor and move to the correct device
    if not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=dtype).to(DEVICE)
    else:
        K = K.to(dtype=dtype, device=DEVICE)

    kappas = []

    for n in tqdm(ns):
        # Extract the submatrix and add ridge regularization
        K_n = K[:n, :n] + ridge * torch.eye(n, dtype=dtype, device=DEVICE)

        # Compute the inverse
        K_n_inv = torch.linalg.inv(K_n)

        # Compute the trace and append the kappa value
        trace_inv = torch.trace(K_n_inv).item()  # Convert to Python scalar
        kappas.append(trace_inv ** -1)

    return torch.tensor(kappas, dtype=dtype).cpu().numpy()

def kernel_eigenvector_weights(K, Y, min_eigenval_threshold=1e-8):
    if len(Y.shape) == 1:
        Y = Y[None,:]

    K = ensure_torch(K)
    Y = ensure_torch(Y).T
    n, _ = K.shape
    n_Ys = Y.shape[1]

    eigenvals, eigenvecs = torch.linalg.eigh(K / n)
    eigenvals = eigenvals[::-1]
    eigenvecs = eigenvecs[:, ::-1]

    mode_weights = (eigenvecs.T @ Y) ** 2

    valid_indices = eigenvals > min_eigenval_threshold
    filtered_eigenvals = eigenvals[valid_indices]

    geom_mean_eigenvals = []

    for i in range(n_Ys):
        filtered_weights = mode_weights[valid_indices][:,i]

        log_geom_mean_eigenval = (torch.log(filtered_eigenvals) * filtered_weights).sum() / filtered_weights.sum()
        geom_mean_eigenval = torch.exp(log_geom_mean_eigenval).item()
        geom_mean_eigenvals.append(geom_mean_eigenval)

    eigenvals = ensure_numpy(eigenvals)
    mode_weights = ensure_numpy(mode_weights)
    geom_mean_eigenvals = np.array(geom_mean_eigenvals)

    return {
        'eigenvals': eigenvals,
        'mode_weights': mode_weights,
        'geom_mean_eigenval': geom_mean_eigenvals
        }

