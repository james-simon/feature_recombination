import numpy as np
import torch
from tqdm import tqdm

from .utils.general import ensure_numpy, ensure_torch

class Kernel:

    def __init__(self, X, device=None):
        self.X = ensure_torch(X)
        self.K = None
        self.eigvals = None
        self.eigvecs = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def set_K(self, K):
        self.K = K
        self.eigvals = None
        self.eigvecs = None

    def krr(self, y, n_train, ridge=0, shuffle=True, K_override=None):
        K = self.K if K_override is None else K_override
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
        return ensure_torch(dX)
    
    def compute_learning_curve(self, ns, n_test: int, Y, ridge=0):
        train_mses, test_mses, test_lrns = np.zeros_like(ns).T, np.zeros_like(ns).T, np.zeros_like(ns).T

        for index, n_train in tqdm(enumerate(ns), total=len(ns)):
            train_mse, test_mse, test_lrn = self.krr(K_override=self.K[:n_train+n_test,:n_train+n_test], y=Y[:n_train+n_test], n_train=n_train, ridge=ridge)

            train_mses[index].append(train_mse)
            test_mses[index].append(test_mse)
            test_lrns[index].append(test_lrn)

        return {
            'train_mse': train_mses,
            'test_mse': test_mses,
            'test_lrn': test_lrns
        }

    def kappa_trace(self, ns, ridge, dtype=torch.float64):
        """
        Compute the experimental kappa values for a range of sizes with ridge regularization.

        Args:
            K (Kernel class): Kernel from which the kernel matrix can be obtained.
            ns (list of int): List of sizes to compute kappa for.
            ridge (float): Regularization parameter.
            dtype (torch.dtype): Data type, e.g., torch.float32 or torch.float64.

        Returns:
            np.ndarray: Array of kappa values for each size in ns.
        """
        
        assert self.K is not None, "Kernel matrix K must be provided."

        kappas = []

        for n in tqdm(ns):
            # Extract the submatrix and add ridge regularization
            K_n = self.K[:n, :n] + ridge * torch.eye(n, dtype=dtype, device=self.device)

            # Compute the inverse
            K_n_inv = torch.linalg.inv(K_n)

            # Compute the trace and append the kappa value
            trace_inv = torch.trace(K_n_inv).item()  # Convert to Python scalar
            kappas.append(trace_inv ** -1)

        return torch.tensor(kappas, dtype=dtype).cpu().numpy()

    def kernel_eigenvector_weights(self, Y, min_eigenval_threshold=1e-8):
        if len(Y.shape) == 1:
            Y = Y[None,:]

        Y = ensure_torch(Y).T
        n, _ = self.K.shape
        n_Ys = Y.shape[1]

        eigenvals, eigenvecs = torch.linalg.eigh(self.K / n)
        eigenvals = torch.flip(eigenvals, dims=[0])
        eigenvecs = torch.flip(eigenvecs, dims=[1])

        mode_weights = (eigenvecs.T @ Y) ** 2

        valid_indices = eigenvals > min_eigenval_threshold
        filtered_eigenvals = eigenvals[valid_indices]

        geom_mean_eigenvals = []
        median_eigenvals = []

        for i in range(n_Ys):
            filtered_weights = mode_weights[valid_indices][:,i]

            log_geom_mean_eigenval = (torch.log(filtered_eigenvals) * filtered_weights).sum() / filtered_weights.sum()
            geom_mean_eigenval = torch.exp(log_geom_mean_eigenval).item()
            geom_mean_eigenvals.append(geom_mean_eigenval)

            cumulative_weights = torch.cumsum(mode_weights.flip(0), dim=0).flip(0)
            half_crossing_index = (cumulative_weights > 0.5).nonzero(as_tuple=True)[0][-1].item()
            median_eigenval = eigenvals[half_crossing_index].item()
            median_eigenvals.append(median_eigenval)

        eigenvals = ensure_numpy(eigenvals)
        mode_weights = ensure_numpy(mode_weights)
        geom_mean_eigenvals = np.array(geom_mean_eigenvals)

        return {
            'eigenvals': eigenvals,
            'mode_weights': mode_weights,
            'geom_mean_eigenvals': geom_mean_eigenvals,
            'median_eigenvals': median_eigenvals
            }
    
    @staticmethod
    def get_level_coeff_fn(bandwidth, data_eigvals):
        return 1/np.e
        # raise NotImplementedError

    def kernel_function_projection(self, functions):
        # functions.shape should be (samples, nfuncs)
        eigvals, eigvecs = self.eigendecomp()
        eigvals = eigvals.flip(0,)
        functions /= torch.linalg.norm(functions, axis=0)
        overlaps = (eigvecs.T @ functions)**2
        # overlap has shape (neigvecs, nfuncs)
        nfuncs = functions.shape[1]
        cdfs = overlaps.flip(0,).cumsum(axis=0)

        quartiles = np.zeros((nfuncs, 3))
        for i in range(nfuncs):
            cdf = cdfs[:, i]
            quartiles[i, 0] = eigvals[cdf >= 0.25][0]
            quartiles[i, 1] = eigvals[cdf >= 0.5][0]
            quartiles[i, 2] = eigvals[cdf >= 0.75][0]
        cdfs = cdfs.flip(0,)

        return ensure_numpy(overlaps.T), ensure_numpy(cdfs.T), quartiles

#kernels TODO: replace self.K with self.K = self.get_K(args, chunking_on, chunk_size)
#also need to potentially convert to numpy for chunking, then back to torch for final K matrix(?) ie Xi = ensure_numpy(Xi)
"""
sample chunking method
# Compute the kernel matrix in chunks
    for i in tqdm(range(0, n_samples1, chunk_size)):
        for j in range(0, n_samples2, chunk_size):
            X1_chunk = X1[i:i + chunk_size]
            X2_chunk = X2[j:j + chunk_size]
            diffs = X1_chunk[:, None, :] - X2_chunk[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            K[i:i + chunk_size, j:j + chunk_size] = np.exp(-0.5 * (dists / width) ** 2)
"""
class ExponentialKernel(Kernel):

    def __init__(self, X, bandwidth):
        super().__init__(X)
        K_lin = self.X @ self.X.T
        self.K = torch.exp(K_lin / bandwidth ** 2)


class GaussianKernel(Kernel):

    def __init__(self, X, bandwidth):
        super().__init__(X)
        dX = self.get_dX()
        assert torch.all(dX.T == dX), "dX must be symmetric"
        assert torch.all(dX >= 0), "dX must be symmetric"
        self.bandwidth = bandwidth
        self.K = torch.exp(-0.5 * (self.get_dX() / bandwidth) ** 2)

    @staticmethod
    def get_level_coeff_fn(bandwidth, data_eigvals):
        q = data_eigvals.sum().item()
        precision = 1 / bandwidth**2
        def eval_level_coeff(k):
            return precision**k * np.exp(-precision*q)
        return eval_level_coeff


class LaplaceKernel(Kernel):

    def __init__(self, X, bandwidth):
        super().__init__(X)
        self.bandwidth = bandwidth
        self.K = torch.exp(-self.get_dX() / bandwidth)

    @staticmethod
    def get_level_coeff_fn(bandwidth, data_eigvals):
        q = data_eigvals.sum().item()
        s = bandwidth
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

#convert from one X to X1 X2?
class RandomFeatureKernel(Kernel):

    def __init__(self, X, nonlinearity=None, num_features=1000):
        super().__init__(X)
        self.nonlinearity = nonlinearity
        self.num_features = num_features
        self.randomize_features(num_features)

    def randomize_features(self, num_features=None):
        if num_features is None:
            num_features = self.num_features
        d = self.X.shape[1]
        W = ensure_torch(torch.normal(0, 1/np.sqrt(d), size=(d, num_features)))
        features = self.X @ W
        if self.nonlinearity:
            features = self.nonlinearity(features)
        self.set_K(features @ features.T)