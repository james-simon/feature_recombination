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

def hermite_product(X, exponents):
  n, d = X.shape

  fn_vals = np.ones(n)
  for idx in exponents:
    fn_vals *= hermite(exponents[idx], X[:, idx])

  return fn_vals


