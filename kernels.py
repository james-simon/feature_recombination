import numpy as np
import torch
from math import factorial
from utils import ensure_torch, ensure_numpy


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
    y_hat_test = y_hat[n_train:]

    return (y_hat_test, y_test), (y_hat_train, y_train)


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


def kernel_hermite_overlap_estimation(kernel, H):
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


class Kernel:

    def __init__(self, X):
        assert X.ndim == 2
        self.X = ensure_torch(X)
        self.K = None
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
    
    def serialize(self):
        return {
            "X": ensure_numpy(self.X),
            "K": ensure_numpy(self.K),
            "eigvals": ensure_numpy(self.eigvals),
            "eigvecs": ensure_numpy(self.eigvecs)
        }
    
    @classmethod
    def deserialize(cls, data):
        obj = cls(**data)
        obj.K = ensure_torch(data["K"])
        if data["eigvals"] is not None:
            obj.eigvals = ensure_torch(data["eigvals"])
        if data["eigvecs"] is not None:
            obj.eigvecs = ensure_torch(data["eigvecs"])
        return obj


class ExponentialKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        K_lin = self.X @ self.X.T
        self.kernel_width = kwargs["kernel_width"]
        self.K = torch.exp(K_lin / (self.kernel_width ** 2))

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        effective_bandwidth = kwargs["kernel_width"]**2 / q
        def eval_level_coeff(ell):
            return effective_bandwidth**(-ell)
        return eval_level_coeff
    
    def serialize(self):
        data = super().serialize()
        data["kernel_width"] = self.kernel_width
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = super().deserialize(data)
        obj.kernel_width = data["kernel_width"]
        return obj


class DotProdKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        K_lin = self.X @ self.X.T
        self.kernel_width = kwargs["kernel_width"]
        self.coeffs = kwargs["coeffs"]
        K_lin /= (self.kernel_width ** 2)
        for k, coeff in enumerate(self.coeffs):
            term = (coeff / factorial(k)) * K_lin**k
            if k == 0:
                self.K = term
            else:
                self.K += term

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        effective_bandwidth = kwargs["kernel_width"]**2 / q
        coeffs = kwargs["coeffs"]
        def eval_level_coeff(ell):
            if ell >= len(coeffs):
                return 0
            return coeffs[ell] * effective_bandwidth**(-ell)
        return eval_level_coeff
    
    def serialize(self):
        data = super().serialize()
        data["kernel_width"] = self.kernel_width
        data["coeffs"] = self.coeffs
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = super().deserialize(data)
        obj.kernel_width = data["kernel_width"]
        obj.coeffs = data["coeffs"]
        return obj


class GaussianKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        self.kernel_width = kwargs["kernel_width"]
        self.K = torch.exp(-0.5 * (self.get_dX() / self.kernel_width) ** 2)

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        effective_bandwidth = kwargs["kernel_width"]**2 / q
        def eval_level_coeff(ell):
            return np.exp(-1/effective_bandwidth) * effective_bandwidth**(-ell)
        return eval_level_coeff
    
    def serialize(self):
        data = super().serialize()
        data["kernel_width"] = self.kernel_width
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = super().deserialize(data)
        obj.kernel_width = data["kernel_width"]
        return obj


class LaplaceKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        self.kernel_width = kwargs["kernel_width"]
        self.K = torch.exp(-self.get_dX() / (np.sqrt(2)*self.kernel_width))

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        s = kwargs["kernel_width"]
        s_eff = s / np.sqrt(q)
        
        def bessel_poly(ell):
            if ell == -1:
                return 1
            y_ell = 0
            for k in range(ell+1):
                term = factorial(ell + k) / (factorial(ell-k)*factorial(k))
                term *= (s_eff/2)**k
                y_ell += term
            return y_ell

        def eval_level_coeff(ell):
            return np.exp(-1/s_eff) * bessel_poly(ell-1) / ((2*s_eff)**ell)
        return eval_level_coeff
    
    def serialize(self):
        data = super().serialize()
        data["kernel_width"] = self.kernel_width
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = super().deserialize(data)
        obj.kernel_width = data["kernel_width"]
        return obj


# class ReluNNGPKernel(Kernel):

#     def __init__(self, X, **kwargs):
#         super().__init__(X)
#         norms = torch.linalg.norm(self.X, dim=-1, keepdim=True)
#         K_norm = norms @ norms.T
#         theta = torch.acos((self.X @ self.X.T / K_norm).clip(-1, 1))
#         angular = torch.sin(theta) + (np.pi - theta)*torch.cos(theta)
#         self.K = 1/(2*np.pi) * K_norm * angular

#     @staticmethod
#     def get_level_coeff_fn(data_eigvals, **kwargs):
#         q = data_eigvals.sum().item()
#         numerator = {
#             0: 1,
#             1: np.pi / 2,
#             2: 1,
#             3: 0,
#             4: 1,
#             5: 0,
#             6: 9,
#             7: 0,
#             8: 225,
#             9: 0,
#             10: 11025,
#         }

#         def eval_level_coeff(ell):
#             assert ell in numerator, f"level coeff {ell} not solved (sorry!)"
#             f = numerator[ell] / q**ell
#             return f * q / (2*np.pi)
#         return eval_level_coeff


# class RandomFeatureKernel(Kernel):

#     def __init__(self, X, **kwargs):
#         super().__init__(X)
#         self.nonlinearity = kwargs["nonlinearity"]
#         self.num_features = kwargs["num_features"]
#         self.randomize_features(self.num_features)

#     def randomize_features(self, num_features=None):
#         if num_features is None:
#             num_features = self.num_features
#         d = self.X.shape[1]
#         W = ensure_torch(torch.normal(0, 1/np.sqrt(d), size=(d, num_features)))
#         features = self.X @ W
#         if self.nonlinearity:
#             features = self.nonlinearity(features)
#         self.set_K(features @ features.T)
