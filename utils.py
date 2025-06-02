import numpy as np
import torch
import math


def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def ensure_torch(x, dtype=torch.float64):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=dtype).to(DEVICE)
    return x.to(dtype).to(DEVICE)


def get_matrix_hermites(X, monomials):
    N, _ = X.shape
    U, _, _ = torch.linalg.svd(X, full_matrices=False)
    X_norm = np.sqrt(N) * U

    hermites = {
        1: lambda x: x,
        2: lambda x: x**2 - 1,
        3: lambda x: x**3 - 3*x,
        4: lambda x: x**4 - 6*x**2 + 3,
        5: lambda x: x**5 - 10*x**3 + 15*x,
        6: lambda x: x**6 - 15*x**4 + 45*x**2 - 15,
        7: lambda x: x**7 - 21*x**5 + 105*x**3 - 105*x,
        8: lambda x: x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105,
        9: lambda x: x**9 - 36*x**7 + 378*x**5 - 1260*x**3 + 945*x,
        10: lambda x: x**10 - 45*x**8 + 630*x**6 - 3150*x**4 + 4725*x**2 - 945,
    }

    H = ensure_torch(torch.zeros((N, len(monomials))))
    for i, monomial in enumerate(monomials):
        h = ensure_torch(torch.ones(N) / np.sqrt(N))
        for d_i, exp in monomial.items():
            Z = np.sqrt(math.factorial(exp))
            h *= hermites[exp](X_norm[:, d_i]) / Z
        H[:, i] = h
    return H