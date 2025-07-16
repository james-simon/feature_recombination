import numpy as np
import torch
from dataclasses import dataclass, field, asdict
import json
import base64


@dataclass(frozen=True)
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
    beta: float             = 1.1
    noise_var: float        = 0.
    # If using natural image data, set these
    zca_strength: float     = 5e-3
    
    def save(self, filepath):
        """Save hyperparameters to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    def generate_filepath(self):
        fp = f"{self.kernel_name}-kw:{self.kernel_width}"
        if self.dataset == "gaussian":
            fp += f"-data:{self.data_dim}:{self.data_eigval_exp}:{self.beta}:{self.noise_var:.3f}"
        else:
            fp += f"-zca:{self.zca_strength}"
        return fp


def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ensure_torch(x, dtype=torch.float32):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype, device=DEVICE)
    return x.to(dtype=dtype, device=DEVICE)


def get_data_eigvals(X):
    N, _ = X.shape
    S = torch.linalg.svdvals(X)
    # to make norm(x)~1 on average (f)
    X *= torch.sqrt(N / (S**2).sum())
    data_eigvals = S**2 / (S**2).sum()
    return data_eigvals

def grab_eigval_distributions(X):
    # N, d = X.shape

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    #each column is an vector corresponding to the i-th eigenvalue
    eigenvector_array = U @ torch.diag(S)# * np.sqrt(N)

    return eigenvector_array, Vt

def cossim_per_eigvec(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    return np.sum(v1_norm * v2_norm, axis=1)

#helper fns for experimental tests (TF decomp)
def find_iterables(d, no_list=["H", "y"]):
    return {k: v for k, v in d.items() if (isinstance(v, (list, np.ndarray)) and k not in no_list)}

def find_statics(d):
    return {k: v for k, v in d.items() if not isinstance(v, (list, np.ndarray))}