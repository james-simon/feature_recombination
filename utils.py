import numpy as np
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


