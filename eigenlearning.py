import scipy


def eigenlearning(n, eigvals, eigcoeffs, ridge=0, noise_var=0):

    def solve_kappa(n, eigvals, ridge):
        conservation_law = lambda kap: (eigvals/(eigvals+kap)).sum() + ridge/kap - n
        kappa = scipy.optimize.bisect(conservation_law, 1e-25, 1e10, maxiter=128)
        return kappa

    kappa = solve_kappa(n, eigvals, ridge)
    learnabilities = eigvals / (eigvals + kappa)
    e0 = n / (n - (learnabilities**2).sum())
    test_mse = e0 * (((1-learnabilities)**2 * eigcoeffs**2).sum() + noise_var)
    train_mse = (ridge / (n * kappa))**2 * test_mse
    L = (eigcoeffs**2 * learnabilities).sum() / (eigcoeffs**2).sum()

    return {
        "kappa": kappa,
        "learnabilities": learnabilities,
        "overfitting_coeff": e0,
        "target_learnability": L,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }