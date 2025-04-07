import numpy as np
import torch
from utils.general import ensure_torch


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

    def krr(self, y, n_train, ridge=0, shuffle=True):
        K = self.K
        y = ensure_torch(y)
        if shuffle:
            idxs = ensure_torch(torch.randperm(K.shape[0], dtype=torch.int32))
            K = K[idxs, idxs]
            y = y[idxs]

        K_train, K_test = K[:n_train, :n_train], K[:, :n_train]
        y_train, y_test = y[:n_train], y[n_train:]

        if ridge == 0:
            alpha = torch.linalg.lstsq(K_train, y_train)
        else:
            eye = ensure_torch(torch.eye(n_train))
            alpha = torch.linalg.lstsq(K_train + ridge * eye, y_train)

        y_hat = K_test @ alpha.solution
        # Train error
        y_hat_train = y_hat[:n_train]
        train_mse = ((y_train - y_hat_train) ** 2).mean(axis=0)

        # Test error
        y_hat_test = y_hat[n_train:]
        test_mse = ((y_test - y_hat_test) ** 2).mean(axis=0)

        test_lrn = (y_test * y_hat_test).mean(axis=0) / (y_test ** 2).mean(axis=0)
        test_lrn = test_lrn

        return train_mse, test_mse, test_lrn

    def estimate_kappa(self, n, ridge=0, shuffle=True):
        K = self.K
        if shuffle:
            idxs = ensure_torch(torch.randperm(K.shape[0], dtype=torch.int32))
            K = K[idxs, idxs]

        if ridge == 0:
            K_n = K[:n, :n]
        else:
            eye = ensure_torch(torch.eye(n))
            K_n = K[:n, :n] + ridge * eye
        kappa = 1 / torch.trace(torch.linalg.pinv(K_n)).item()
        return kappa

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


class ExponentialKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        K_lin = self.X @ self.X.T
        self.K = torch.exp(K_lin / kwargs["bandwidth"] ** 2)


class GaussianKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        dX = self.get_dX()
        self.bandwidth = kwargs["bandwidth"]
        self.K = torch.exp(-0.5 * (self.get_dX() / self.bandwidth) ** 2)

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        precision = 1 / kwargs["bandwidth"]**2
        norm = kwargs["avg_norm"] if "avg_norm" in kwargs else 1
        def eval_level_coeff(k):
            return (norm*precision)**k * np.exp(-precision*q)
        return eval_level_coeff


class LaplaceKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        self.bandwidth = kwargs["bandwidth"]
        self.K = torch.exp(-self.get_dX() / self.bandwidth)

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        q = data_eigvals.sum().item()
        s = kwargs["bandwidth"]
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
