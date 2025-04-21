import torch
import numpy as np
from utils.stats import grab_eigval_distributions
from utils.general import ensure_torch

def eigvecs_to_independent(eigenvectors, bsz=None, rng = None, to_torch=True):
    rng = np.random.default_rng() if rng is None else rng
    n, dim = eigenvectors.shape #n = num datapoints
    bsz = n if bsz is None else bsz
    independent_components = torch.zeros((bsz, dim))

    for i in range(bsz):
        # For each column, randomly select one value from the n rows
        random_rows = rng.integers(0, n, size=dim)
        independent_components[i] = eigenvectors[random_rows, np.arange(dim)]

    return ensure_torch(independent_components) if to_torch else independent_components

def independentize_data(X, bsz=1, rng=None):
    eigenvectors, Vt = grab_eigval_distributions(X)
    independent_data = eigvecs_to_independent(eigenvectors, bsz, rng)
    return independent_data @ Vt