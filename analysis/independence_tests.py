import torch
import numpy as np

def grab_eigval_distributions(X):
    N, d = X.shape

    U, S, Vt = torch.linalg.svd(X, full_matrices=False)

    #each column is an vector corresponding to the i-th eigenvalue
    eigenvector_array = np.sqrt(N) * U @ torch.diag(S)

    return eigenvector_array, Vt

def independent_distributions(eigenvectors, bsz=1, rng = None):
    rng = np.random.default_rng() if rng is None else rng
    m, n = eigenvectors.shape
    independent_components = torch.zeros((bsz, n))

    for i in range(bsz):
        # For each column, randomly select one value from the m rows
        random_rows = rng.integers(0, m, size=n)
        independent_components[i] = eigenvectors[random_rows, np.arange(n)]

    return independent_components

def generate_independent_data(X, bsz=1, rng=None):
    eigenvectors, _ = grab_eigval_distributions(X)
    independent_data = independent_distributions(eigenvectors, bsz, rng)
    return independent_data