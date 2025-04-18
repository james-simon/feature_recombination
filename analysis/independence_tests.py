import torch
import numpy as np
from utils.stats import grab_eigval_distributions
from utils.general import ensure_torch

def eigvecs_to_independent(eigenvectors, bsz=1, rng = None):
    rng = np.random.default_rng() if rng is None else rng
    m, n = eigenvectors.shape #m = num datapoints, m=n
    independent_components = torch.zeros((bsz, n))

    for i in range(bsz):
        # For each column, randomly select one value from the m rows
        random_rows = rng.integers(0, m, size=n)
        independent_components[i] = eigenvectors[random_rows, np.arange(n)]

    return ensure_torch(independent_components)

def independentize_data(X, bsz=1, rng=None):
    eigenvectors, Vt = grab_eigval_distributions(X)
    independent_data = eigvecs_to_independent(eigenvectors, bsz, rng)
    return ensure_torch(independent_data) @ Vt