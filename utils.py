import numpy as np
import torch


def ensure_numpy(x, dtype=np.float64):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def ensure_torch(x, dtype=torch.float32):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype, device=DEVICE)
    return x.to(dtype=dtype, device=DEVICE)
