import numpy as np
import torch
import math
from utils import ensure_torch, ensure_numpy


class Kernel:

    def __init__(self, X):
        assert X.ndim == 2
        self.X = ensure_torch(X)
        self.K = None
        self.eigvals = None
        self.eigvecs = None

    def set_K(self, K):
        self.K = K
        self.eigvals = None
        self.eigvecs = None

    def eigenvals(self):
        if self.eigvals is None:
            n = self.X.shape[0]
            self.eigvals = torch.linalg.eigvalsh(self.K / n).flip((0,))
        return self.eigvals

    def eigendecomp(self):
        if self.eigvecs is None:
            n = self.X.shape[0]
            eigvals, eigvecs = torch.linalg.eigh(self.K / n)
            self.eigvals = eigvals.flip((0,))
            self.eigvecs = eigvecs.flip((1,))
        return self.eigvals, self.eigvecs

    def get_dX(self):
        if not torch.cuda.is_available():
            return torch.linalg.norm(self.X[:, None, :] - self.X[None, :, :], axis=-1)
        gpu_mem_avail, _ = torch.cuda.mem_get_info()
        N, d = self.X.shape
        step = int(0.1 * gpu_mem_avail / (N * d * 4))
        dX = torch.zeros((N, N))
        for i in range(0, N, step):
            dX[i:i+step, :] = torch.linalg.norm(self.X[i:i+step, None, :] - self.X[None, :, :], axis=-1)
        dX = ensure_torch(dX)
        assert torch.all(dX.T == dX), "dX must be symmetric"
        assert torch.all(dX >= 0), "dX must be nonnegative"
        return dX


def krr(K, y, n_train, n_test, ridge=0):
    n_tot = K.shape[0]
    assert n_train + n_test <= n_tot
    train_slc = torch.randperm(n_tot - n_test)[:n_train]
    test_slc = torch.arange(n_tot - n_test, n_tot)
    slc = torch.cat([train_slc, test_slc])
    K = K[slc[:, None], slc[None, :]]
    y = y[slc]

    K_train, K_test = K[:n_train, :n_train], K[:, :n_train]
    y_train, y_test = y[:n_train], y[n_train:]

    if ridge == 0:
        alpha = torch.linalg.lstsq(K_train, y_train)
    else:
        eye = ensure_torch(torch.eye(n_train))
        alpha = torch.linalg.lstsq(K_train + ridge * eye, y_train)

    y_hat = K_test @ alpha.solution
    
    y_hat_train = y_hat[:n_train]
    train_mse = ((y_train - y_hat_train) ** 2).mean(axis=0)
    train_mse = train_mse.sum().item()

    y_hat_test = y_hat[n_train:]
    test_mse = ((y_test - y_hat_test) ** 2).mean(axis=0)
    test_mse = test_mse.sum().item()

    return train_mse, test_mse, y_hat


def estimate_kappa(kernel, n, ridge=0):
    K = kernel.K
    n_tot = K.shape[0]
    slc = torch.randperm(n_tot)[:n]
    K_n = K[slc[:, None], slc[None, :]]

    if ridge != 0:
        eye = ensure_torch(torch.eye(n))
        K_n = K_n + ridge * eye
    kappa = 1 / torch.trace(torch.linalg.pinv(K_n)).item()
    return kappa


def project_hermites(kernel, H):
    # H.shape should be (samples, nhermites)
    assert kernel.eigvals is not None, "Call eigendecomp() first"
    eigvals = kernel.eigvals.flip(0,)
    H = ensure_torch(H)
    H /= torch.linalg.norm(H, axis=0)
    overlaps = (kernel.eigvecs.T @ H)**2
    # overlap has shape (neigvecs, nhermites)
    nhermites = H.shape[1]
    cdfs = overlaps.flip(0,).cumsum(axis=0)

    quartiles = np.zeros((nhermites, 3))
    for i in range(nhermites):
        cdf = cdfs[:, i]
        quartiles[i, 0] = eigvals[cdf >= 0.25][0]
        quartiles[i, 1] = eigvals[cdf >= 0.5][0]
        quartiles[i, 2] = eigvals[cdf >= 0.75][0]
    cdfs = cdfs.flip(0,)

    return ensure_numpy(overlaps.T), ensure_numpy(cdfs.T), quartiles


class ExponentialKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        K_lin = self.X @ self.X.T
        self.kernel_width = kwargs["kernel_width"]
        self.K = torch.exp(K_lin / (self.kernel_width ** 2))

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        precision = 1 / kwargs["kernel_width"]**2
        def eval_level_coeff(k):
            return (precision)**k
        return eval_level_coeff


class DotProdKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        K_lin = self.X @ self.X.T
        self.kernel_width = kwargs["kernel_width"]
        self.coeffs = kwargs["coeffs"]
        K_lin /= (self.kernel_width ** 2)
        for k, coeff in enumerate(self.coeffs):
            term = (coeff / math.factorial(k)) * K_lin**k
            if k == 0:
                self.K = term
            else:
                self.K += term

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        precision = 1 / kwargs["kernel_width"]**2
        coeffs = kwargs["coeffs"]
        def eval_level_coeff(k):
            if k >= len(coeffs):
                return 0
            return coeffs[k] * (precision)**k
        return eval_level_coeff


class GaussianKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        self.kernel_width = kwargs["kernel_width"]
        self.K = torch.exp(-0.5 * (self.get_dX() / self.kernel_width) ** 2)

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        precision = 1 / kwargs["kernel_width"]**2
        def eval_level_coeff(k):
            return (precision)**k * np.exp(-precision*q)
        return eval_level_coeff


class LaplaceKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        self.kernel_width = kwargs["kernel_width"]
        self.K = torch.exp(-self.get_dX() / self.kernel_width)

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        s = kwargs["kernel_width"]
        numerator = {
            0: 1,
            1: np.sqrt(2),
            2: (2 * np.sqrt(q) + np.sqrt(2) * s),
            3: (2 * np.sqrt(2) * q + 6 * np.sqrt(q) * s + 3 * np.sqrt(2) * s**2),
            4: (4 * q ** (3/2) + 12 * np.sqrt(2) * q * s + 30 * np.sqrt(q) * s**2 + 15 * np.sqrt(2) * s**3),
            5: (4 * np.sqrt(2) * q**2 + 40 * q ** (3/2) * s + 90 * np.sqrt(2) * q * s**2 + 210 * np.sqrt(q) * s**3 \
                + 105 * np.sqrt(2) * s**4),
            6: (8 * q ** (5/2) + 60 * np.sqrt(2) * q**2 * s + 420 * q ** (3/2) * s**2 + 840 * np.sqrt(2) * q * s**3 \
                + 1890 * np.sqrt(q) * s**4 + 945 * np.sqrt(2) * s**5),
            7: (8 * np.sqrt(2) * q**3 + 168 * q ** (5/2) * s + 840 * np.sqrt(2) * q**2 * s**2 \
                + 5040 * q ** (3/2) * s**3 + 9450 * np.sqrt(2) * q * s**4 + 20790 * np.sqrt(q) * s**5 \
                + 10395 * np.sqrt(2) * s**6),
            8: (16 * q ** (7/2) + 224 * np.sqrt(2) * q**3 * s + 3024 * q ** (5/2) * s**2 \
                + 12600 * np.sqrt(2) * q**2 * s**3 + 69300 * q ** (3/2) * s**4 + 124740 * np.sqrt(2) * q * s**5 \
                + 270270 * np.sqrt(q) * s**6 + 135135 * np.sqrt(2) * s**7),
            9: (16 * np.sqrt(2) * q**4 + 576 * q ** (9/2) * s + 5040 * np.sqrt(2) * q**3 * s**2 \
                + 55440 * q ** (5/2) * s**3 + 207900 * np.sqrt(2) * q**2 * s**4 + 1081080 * q ** (3/2) * s**5 \
                + 1891890 * np.sqrt(2) * q * s**6 + 4054050 * np.sqrt(q) * s**7 + 2027025 * np.sqrt(2) * s**8),
            10: (32 * q ** (9/2) + 720 * np.sqrt(2) * q**4 * s + 15840 * q ** (7/2) * s**2 \
                 + 1081080 * np.sqrt(2) * q**3 * s**3 + 3783780 * q ** (5/2) * s**4 \
                 + 18918900 * np.sqrt(2) * q**2 * s**5 + 18918900 * q ** (3/2) * s**6 \
                 + 32432400 * np.sqrt(2) * q * s**7 + 68918850 * np.sqrt(q) * s**8 + 34459425 * np.sqrt(2) * s**9),
        }

        def eval_level_coeff(k):
            assert k in numerator, f"k={k} level coeff not solved (sorry!)"
            f = numerator[k] / (q**(-1/2) * (2*s*q)**k)
            return f * np.exp(-np.sqrt(2*q)/s)
        return eval_level_coeff


class ReluNNGPKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        norms = torch.linalg.norm(self.X, dim=-1, keepdim=True)
        K_norm = norms @ norms.T
        d = self.X.shape[1]
        theta = torch.acos((self.X @ self.X.T / K_norm).clip(-1, 1))
        angular = torch.sin(theta) + (np.pi - theta)*torch.cos(theta)
        self.K = 1/(2*np.pi) * K_norm * angular

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        numerator = {
            0: 1,
            1: np.pi / 2,
            2: 1,
            3: 0,
            4: 1,
            5: 0,
            6: 9,
            7: 0,
            8: 225,
            9: 0,
            10: 11025,
        }

        def eval_level_coeff(k):
            assert k in numerator, f"k={k} level coeff not solved (sorry!)"
            f = numerator[k] / q**k
            return f * q / (2*np.pi)
        return eval_level_coeff


class RandomFeatureKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        self.nonlinearity = kwargs["nonlinearity"]
        self.num_features = kwargs["num_features"]
        self.randomize_features(self.num_features)

    def randomize_features(self, num_features=None):
        if num_features is None:
            num_features = self.num_features
        d = self.X.shape[1]
        W = ensure_torch(torch.normal(0, 1/np.sqrt(d), size=(d, num_features)))
        features = self.X @ W
        if self.nonlinearity:
            features = self.nonlinearity(features)
        self.set_K(features @ features.T)
