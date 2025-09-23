import numpy as np
import torch
from math import factorial
from utils import ensure_torch, ensure_numpy


def krr(K, y, n_train, n_test, ridge=0, trial=None):
    n_tot = K.shape[0]
    assert n_train + n_test <= n_tot
    if trial is not None:
        train_slc = torch.arange(n_train) + trial * n_train
        train_slc = train_slc % (n_tot - n_test)
    else:
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


def krr_loop(K, y, n_train, n_test, ntrials, ridge=0):
    
    def shuffle_indices(n_tot, K, y):
        slc = torch.randperm(n_tot)
        K = K[slc[:, None], slc[None, :]]
        y = y[slc]
        return K, y
    
    n_tot = K.shape[0]
    n_per_trial = n_train + n_test
    assert n_per_trial <= n_tot
    
    cur_idx = 0
    train_mses = []
    test_mses = []
    for _ in range(ntrials):
        if cur_idx + n_per_trial > n_tot:
            K, y = shuffle_indices(n_tot, K, y)
            cur_idx = 0
        end_idx_train = cur_idx + n_train
        end_idx_test = cur_idx + n_per_trial
        
        K_train = K[cur_idx:end_idx_train, cur_idx:end_idx_train]
        K_test = K[cur_idx:end_idx_test, cur_idx:end_idx_train]
        y_train = y[cur_idx:end_idx_train]
        y_test = y[end_idx_train:end_idx_test]

        if ridge == 0:
            alpha = torch.linalg.lstsq(K_train, y_train)
        else:
            eye = ensure_torch(torch.eye(n_train))
            alpha = torch.linalg.lstsq(K_train + ridge * eye, y_train)

        y_hat = K_test @ alpha.solution
        y_hat_train = y_hat[:n_train]
        y_hat_test = y_hat[n_train:]

        train_mse = ((y_hat_train - y_train)**2).mean(axis=-1).item()
        test_mse = ((y_hat_test - y_test)**2).mean(axis=-1).item()

        train_mses.append(train_mse)
        test_mses.append(test_mse)

        cur_idx += n_per_trial

    train_mses = np.array(train_mses)
    test_mses = np.array(test_mses)
    return test_mses, train_mses


def estimate_kappa(kernel, n, ridge=0):
    K = kernel.K
    n_tot = K.shape[0]
    slc = torch.randperm(n_tot)[:n]
    K_n = K[slc[:, None], slc[None, :]]

    if ridge != 0:
        eye = ensure_torch(torch.eye(n))
        K_n = K_n + ridge * eye
    kappa = 1 / torch.trace(torch.linalg.inv(K_n)).item()
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


class ReluNNGPKernel(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        w_var, b_var = ReluNNGPKernel.parse_kwargs(kwargs)
        self.w_var = w_var
        self.b_var = b_var

        norms = torch.linalg.norm(self.X, dim=-1, keepdim=True)
        p = b_var + w_var * norms**2
        normalizer = (p @ p.T).sqrt()

        first_layer_nngp = ((w_var * self.X @ self.X.T) + b_var)
        theta = torch.acos((first_layer_nngp / normalizer).clip(-1, 1))
        self.K = (w_var * normalizer * (torch.sin(theta) + torch.cos(theta) * (np.pi - theta))) / (2*np.pi)

    @staticmethod
    def parse_kwargs(kwargs):
        if "weight_variance" in kwargs and "bias_variance" in kwargs:
            w_var = kwargs["weight_variance"]
            b_var = kwargs["bias_variance"]
        else:
            k_width = kwargs.get("kernel_width", 1.0)
            w_var = np.sqrt((2*np.pi)/(1 + 2*np.pi*k_width))
            b_var = k_width * w_var
        return w_var, b_var
    
    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        w_var, b_var = ReluNNGPKernel.parse_kwargs(kwargs)

        rho = ensure_numpy(data_eigvals).sum()
        q = b_var + w_var * rho
        c0 = b_var / q
        if abs(c0) >= 1.0:
            raise ValueError("|c0|=1 makes higher derivatives singular; ensure w_var>0.")

        pref = w_var * q / (2*np.pi)
        scale = w_var / q
        
        def _poly_add(a, b, sa=1.0, sb=1.0):
            n = max(len(a), len(b))
            out = [0.0]*n
            for i in range(n):
                va = a[i] if i < len(a) else 0.0
                vb = b[i] if i < len(b) else 0.0
                out[i] = sa*va + sb*vb
            return out

        def _poly_eval(cs, c):
            p = 0.0
            for k in reversed(range(len(cs))):
                p = p*c + cs[k]
            return p

        def eval_level_coeff(ell):
            if ell == 0:
                c = float(np.clip(c0, -1.0, 1.0))
                Fk_at_c0 = np.sqrt(max(0.0, 1.0 - c**2)) + (np.pi - np.arccos(c))*c
            elif ell == 1:
                c = float(np.clip(c0, -1.0, 1.0))
                Fk_at_c0 = np.pi - np.arccos(c)
            elif ell == 2:
                Fk_at_c0 = 1.0 / np.sqrt(1.0 - c0**2)
            else:
                Q = [1.0]
                for cur_ell in range(2, ell):
                    term1 = [k*Q[k] for k in range(1, len(Q))]
                    term1 = _poly_add(term1, [0.0, 0.0] + term1, 1.0, -1.0)
                    term2 = [0.0] + list(Q)
                    Q = _poly_add(term1, term2, 1.0, (2*cur_ell - 3))
                Fk_at_c0 = _poly_eval(Q, c0) / (1.0 - c0**2)**(ell - 1.5)
            return pref * (scale**ell) * float(Fk_at_c0)

        return eval_level_coeff
    
    def serialize(self):
        data = super().serialize()
        data["w_var"] = self.w_var
        data["b_var"] = self.b_var
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = super().deserialize(data)
        obj.w_var = data["w_var"]
        obj.b_var = data["b_var"]
        return obj


class ReluNTK(Kernel):

    def __init__(self, X, **kwargs):
        super().__init__(X)
        w_var, b_var = ReluNTK.parse_kwargs(kwargs)
        self.w_var = w_var
        self.b_var = b_var

        norms = torch.linalg.norm(self.X, dim=-1, keepdim=True)
        p = b_var + w_var * norms**2
        if kwargs.get("use_numpy", False):
            p = ensure_numpy(p)
            X = ensure_numpy(self.X)
            normalizer = np.sqrt(p @ p.T)

            first_layer_nngp = (w_var * X @ X.T) + b_var
            theta = np.acos((first_layer_nngp / normalizer).clip(-1, 1))

            second_layer_nngp = (w_var * normalizer * (np.sin(theta) + np.cos(theta) * (np.pi - theta))) / (2*np.pi)
            second_layer_ntk = (w_var * first_layer_nngp * (np.pi - theta)) / (2*np.pi)
        else:
            normalizer = (p @ p.T).sqrt()

            first_layer_nngp = (w_var * self.X @ self.X.T) + b_var
            theta = torch.acos((first_layer_nngp / normalizer).clip(-1, 1))

            second_layer_nngp = (w_var * normalizer * (torch.sin(theta) + torch.cos(theta) * (np.pi - theta))) / (2*np.pi)
            second_layer_ntk = (w_var * first_layer_nngp * (np.pi - theta)) / (2*np.pi)
        
        K = second_layer_nngp + second_layer_ntk
        self.K = ensure_torch(K)
        del normalizer, first_layer_nngp, theta, second_layer_nngp, second_layer_ntk
        torch.cuda.empty_cache()

    @staticmethod
    def parse_kwargs(kwargs):
        if "weight_variance" in kwargs and "bias_variance" in kwargs:
            w_var = kwargs["weight_variance"]
            b_var = kwargs["bias_variance"]
        else:
            k_width = kwargs.get("kernel_width", 1.0)
            w_var = np.sqrt((2*np.pi)/(1 + 2*np.pi*k_width))
            b_var = k_width * w_var
        return w_var, b_var

    @staticmethod
    def get_level_coeff_fn(data_eigvals, **kwargs):
        w_var, b_var = ReluNTK.parse_kwargs(kwargs)

        rho = ensure_numpy(data_eigvals).sum()
        q = b_var + w_var * rho
        c0 = b_var / q
        if abs(c0) >= 1.0:
            raise ValueError("|c0|=1 makes higher derivatives singular; ensure w_var>0.")

        pref = w_var * q / (2*np.pi)
        scale = w_var / q
        
        def _poly_add(a, b, sa=1.0, sb=1.0):
            n = max(len(a), len(b))
            out = [0.0]*n
            for i in range(n):
                va = a[i] if i < len(a) else 0.0
                vb = b[i] if i < len(b) else 0.0
                out[i] = sa*va + sb*vb
            return out

        def _poly_eval(cs, c):
            p = 0.0
            for k in reversed(range(len(cs))):
                p = p*c + cs[k]
            return p

        def eval_level_coeff(ell):
            P = [3.0, 0.0, -2.0]
            if ell == 0:
                c = float(np.clip(c0, -1.0, 1.0))
                Gk_at_c0 = np.sqrt(max(0.0, 1.0 - c**2)) + 2.0*(np.pi - np.arccos(c))*c
            elif ell == 1:
                c = float(np.clip(c0, -1.0, 1.0))
                denom = np.sqrt(max(0.0, 1.0 - c**2))
                Gk_at_c0 = 2.0*(np.pi - np.arccos(c)) + (c/denom if denom > 0 else float('inf'))
            elif ell == 2:
                Gk_at_c0 = _poly_eval(P, c0) / (1.0 - c0*c0)**1.5
            else:
                for cur_ell in range(2, ell):
                    term1 = [k*P[k] for k in range(1, len(P))]
                    term1 = _poly_add(term1, [0.0, 0.0] + term1, 1.0, -1.0)
                    term2 = [0.0] + list(P)
                    P = _poly_add(term1, term2, 1.0, (2*cur_ell - 1))
                Gk_at_c0 = _poly_eval(P, c0) / (1.0 - c0*c0)**(ell - 0.5)
            return pref * (scale**ell) * float(Gk_at_c0)
        
        return eval_level_coeff
    
    def serialize(self):
        data = super().serialize()
        data["w_var"] = self.w_var
        data["b_var"] = self.b_var
        return data
    
    @classmethod
    def deserialize(cls, data):
        obj = super().deserialize(data)
        obj.w_var = data["w_var"]
        obj.b_var = data["b_var"]
        return obj
