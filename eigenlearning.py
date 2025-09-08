import scipy
from utils import ensure_numpy


def compute_learnabilities(n, eigvals, ridge=0):

    def solve_kappa(n, eigvals, ridge):
        eigvals = ensure_numpy(eigvals)
        conservation_law = lambda kap: (eigvals/(eigvals+kap)).sum() + ridge/kap - n
        kappa = scipy.optimize.bisect(conservation_law, 1e-25, 1e10, maxiter=128)
        return kappa

    kappa = solve_kappa(n, eigvals, ridge)
    learnabilities = eigvals / (eigvals + kappa)
    return kappa, learnabilities


def learning_curve(n, learnabilities, eigcoeffs, noise_var=0):
    if len(eigcoeffs) < len(learnabilities):
        learnabilities = learnabilities[:len(eigcoeffs)]
    e0 = n / (n - (learnabilities**2).sum())
    test_mse = e0 * (((1-learnabilities)**2 * eigcoeffs**2).sum() + noise_var)
    return test_mse


def eigenlearning(n, eigvals, eigcoeffs, ridge=0, noise_var=0):
    kappa, learnabilities = compute_learnabilities(n, eigvals, ridge)
    test_mse = learning_curve(n, learnabilities, eigcoeffs, noise_var)
    train_mse = (ridge / (n * kappa))**2 * test_mse
    L = (eigcoeffs**2 * learnabilities).sum() / (eigcoeffs**2).sum()

    return {
        "kappa": kappa,
        "learnabilities": learnabilities,
        "target_learnability": L,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
