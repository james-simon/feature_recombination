import numpy as np
import torch
from dataclasses import dataclass, asdict
import json


@dataclass
class Hyperparams:
    expt_name: str
    dataset: str
    kernel_name: str
    kernel_width: float     = 4
    n_samples: int          = 20_000
    p_modes: int            = 10_000
    # If using synth data, set these
    data_dim: int           = 200
    data_eigval_exp: float  = 1.2
    # If using natural image data, set these
    zca_strength: float     = 5e-3
    
    def save(self, filepath):
        """Save hyperparameters to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    def generate_filepath(self):
        fp = f"{self.kernel_name}-kw:{self.kernel_width}"
        if self.dataset == "gaussian":
            fp += f"-data:{self.data_dim}:{self.data_eigval_exp}"
        else:
            fp += f"-zca:{self.zca_strength}"
        return fp


def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_torch(x, dtype=torch.float32):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype, device=DEVICE)
    return x.to(dtype=dtype, device=DEVICE)