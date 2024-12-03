import numpy as np
import scipy as sp
import torch

def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def ensure_torch(x, dtype=torch.float64):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype).to(DEVICE)
    return x

def rms(x):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.mean(x ** 2))
    elif isinstance(x, torch.Tensor):
        return torch.sqrt(torch.mean(x ** 2))
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

def cos_sim(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def hermite(k, x):
    """Compute the k-th probabilist's Hermite polynomial at x, normalized to have unit norm against the standard normal distribution."""

    return sp.special.hermite(k)(x / 2 ** .5) * 2 ** (-k / 2) / sp.special.factorial(k) ** .5

def monomial_of_array(array, monomial):
  return np.prod([array[idx] ** monomial[idx] for idx in monomial])

def hermite_product(X, exponents):
  n, d = X.shape

  fn_vals = np.ones(n)
  for idx in exponents:
    fn_vals *= hermite(exponents[idx], X[:, idx])

  return fn_vals

def shuffle_indices(**tensors):
    """
    Shuffle multiple numpy arrays along all axes with the same random permutation.

    Args:
        **tensors: Named numpy arrays to shuffle, all of which must have the same shape
                   with all dimensions equal to n.

    Returns:
        dict: A dictionary containing the shuffled numpy arrays, with the same keys as the input.
    """
    # Ensure all tensors are numpy arrays and have compatible shapes
    n = None
    shape = None
    for name, tensor in tensors.items():
        if not isinstance(tensor, np.ndarray):
            raise TypeError(f"{name} must be a numpy array.")

        # Check that all dimensions are equal
        if not all(dim == tensor.shape[0] for dim in tensor.shape):
            raise ValueError(f"All dimensions of {name} must be equal. Found shape {tensor.shape}")

        if n is None:
            n = tensor.shape[0]
            shape = tensor.shape
        else:
            if tensor.shape != shape:
                raise ValueError(f"All arrays must have the same shape. {name} has shape {tensor.shape}.")

    # Generate a fixed random permutation of indices
    permutation = np.random.permutation(n)

    # Create an index tuple for advanced indexing
    idx = np.ix_(*([permutation] * len(shape)))

    # Shuffle all tensors by the same permutation along all axes
    shuffled_tensors = {}
    for name, tensor in tensors.items():
        shuffled_tensors[name] = tensor[idx]

    return shuffled_tensors