import numpy as np
import torch as torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as transformsF, ToTensor
import torch.nn.functional as F

from einops import rearrange
from utils import ensure_torch
from data import get_matrix_hermites
from kernels import GaussianKernel
from feature_decomp import generate_fra_monomials

def get_target_values(Phi, target_coeffs, noise_std):
    Y = None

    if target_coeffs is not None:
        Y = Phi @ target_coeffs

    if noise_std > 0:
        Y += noise_std * np.random.randn(*Y.shape)

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    return Y

def sample_gaussian_data(n_samples, cov_eigvals, target_coeffs=None, noise_std=0):

    Phi = np.random.randn(n_samples, len(cov_eigvals))
    X = Phi * cov_eigvals ** .5

    Y = get_target_values(Phi, target_coeffs, noise_std)

    return X, Y

def sample_uniform_data(n_samples, cov_eigvals, target_coeffs=None, noise_std=0):

    Phi = np.random.uniform(-np.sqrt(3), np.sqrt(3), (n_samples, len(cov_eigvals)))
    X = Phi * cov_eigvals ** .5

    Y = get_target_values(Phi, target_coeffs, noise_std)

    return X, Y

def get_synthetic_X(d=500, N=15000, offset=3, alpha=1.5, **kwargs):
    """
    Powerlaw synthetic data
    """
    data_eigvals = ensure_torch((offset+np.arange(d)) ** -alpha)
    data_eigvals /= data_eigvals.sum()
    X = ensure_torch(torch.normal(0, 1, (N, d))) * torch.sqrt(data_eigvals)
    return X, data_eigvals

def get_v_true(fra_eigvals, ytype='Gaussian', **kwargs):
    assert ytype in ["Gaussian", "Uniform", "Isotropic", "Binarized", "PowerLaw", "OneHot", "NHot"], "Type not found"
    match ytype:
        case "Gaussian":
            return ensure_torch(torch.normal(fra_eigvals, torch.sqrt(fra_eigvals)))
        case "Uniform":
            return ensure_torch(torch.rand(fra_eigvals.shape[0]))
        case "Isotropic":
            return ensure_torch(torch.ones_like(fra_eigvals)/torch.sqrt(torch.tensor(fra_eigvals.shape[0])))
        case "Binarized":
            H = kwargs.get("H", None)
            y_underlying = ensure_torch(torch.randint(low=0, high=2, size=(H.shape[0],)) * 2 - 1)
            v = torch.linalg.lstsq(H, y_underlying).solution
            return ensure_torch(v)
        case "PowerLaw":
            H = kwargs.get("H", None)
            i0 = kwargs.get("vi0", 3)
            alpha = kwargs.get("beta", 1.5)
            pldim = H.shape[1]
            # pldim = kwargs.get("pldim", 400)
            pldecay = ensure_torch((i0+np.arange(pldim)) ** (-alpha/2))
            pldecay /= torch.sqrt((pldecay**2).sum())
            v = pldecay
            # y_underlying = ensure_torch(torch.randint(low=0, high=2, size=(H.shape[0],)) * 2 - 1)
            # y_refactor = y_underlying - H[:, :pldim] @ pldecay
            # v_non_pl = torch.linalg.lstsq(H[:, pldim:], y_refactor).solution
            # v = torch.hstack((pldecay, v_non_pl)).T
            return ensure_torch(v)
        case "OneHot":
            H = kwargs.get("H", None)
            ohindex = kwargs.get("OneHotIndex", None)
            mode_dim = H.shape[1]
            if ohindex == None:
                ohindex = np.random.choice(mode_dim, replace=False)
            v = torch.zeros(mode_dim)
            v[ohindex] = 1
            return ensure_torch(v)
        case "NHot":
            indices = kwargs.get("NHotIndices", 3)
            nhsizes = kwargs.get("NHotSizes", None)
            H = kwargs.get("H", None)
            mode_dim = H.shape[1]
            if type(indices) == int: #if num_indices provided as opposed to the locations of the indices
                indices = np.random.choice(mode_dim, size=indices, replace=False)
            if nhsizes is None:
                nhsizes =  torch.tensor((1.2+np.arange(len(indices))) ** -3, dtype=torch.float32)
                nhsizes /= torch.sqrt((nhsizes**2).sum())
            v = torch.zeros(mode_dim, dtype=torch.float32)
            v[indices] = nhsizes
            return ensure_torch(v)
            
    return None

def get_synthetic_dataset(X=None, data_eigvals=None, ytype="Gaussian", d=500, N=15000, offset=3, alpha=1.5, cutoff_mode=10000,
                          noise_size=0.1, normalized=True, **vargs):
    """
    y_type: One of \"Gaussian\", \"Uniform\", \"Isotropic\", \"Binarized\", "\PowerLaw\", \"OneHot\", \"NHot\"
    noise_size: total noise size of the N-dim target vector y
    """
    if X is None:
        X, data_eigvals = get_synthetic_X(d=d, N=N, offset=offset, alpha=alpha)

    kernel_width = vargs.get("kernel_width", 2)
    kerneltype = vargs.get("kerneltype", GaussianKernel)
    fra_eigvals, monomials = generate_fra_monomials(data_eigvals, cutoff_mode, kerneltype.get_level_coeff_fn(kernel_width=kernel_width, data_eigvals=data_eigvals), kmax=9)
    H = ensure_torch(get_matrix_hermites(X, monomials))
    fra_eigvals = ensure_torch(fra_eigvals)
    v_true = get_v_true(fra_eigvals, ytype, noise_size=noise_size, H=H, **vargs)
    v_true = v_true if not normalized else v_true/torch.linalg.norm(v_true)
    y = ensure_torch(H) @ v_true + ensure_torch(torch.normal(0., noise_size/H.shape[0]**(0.5), (H.shape[0],)))
    return X, y, H, monomials, fra_eigvals, v_true

class SyntheticDataset(Dataset):
    def __init__(self, X=None, data_eigvals=None, ytype="Gaussian", d=500, N=15000, offset=3, alpha=1.5, cutoff_mode=10000,
                          noise_size=0.1, normalized=True, transform=None, **vargs):
        """
        Purely for use with the transform operator. Probably a cleanear way
        """

        self.X, self.y, self.H, self.monomials, self.fra_eigvals, self.v_true = get_synthetic_dataset(X, data_eigvals,
                        ytype, d, N, offset, alpha, cutoff_mode, noise_size, normalized, **vargs)
        self.data_eigvals = data_eigvals
        self.N = N
        self.transform = transform
        if vargs.get("image", False):
            self.X = torch.reshape(self.X, (self.X.shape[0], np.sqrt(self.X.shape[1]).astype(int), np.sqrt(self.X.shape[1]).astype(int)))

    def __len__(self):
        return self.N


    def __getitem__(self, idx):
        
        X = self.X[idx]
        y = self.y[idx]
        if self.transform:
            X = self.transform(X)

        return X, y

class ImageData():
    """
    Get mid-scale image datasets as numpy arrays.
    """

    dataset_dict = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'svhn': torchvision.datasets.SVHN,
        'imagenet32': None,
        'imagenet64': None,
    }

    def __init__(self, dataset_name, data_dir, classes=None, onehot=True, format='NCHW', **kwargs):
        """
        dataset_name (str): one of  'mnist', 'fmnist', 'cifar10', 'cifar100', 'imagenet32', 'imagenet64'
        dataset_dir (str): the directory where the raw dataset is saved
        classes (iterable): a list of groupings of old class labels that each constitute a new class.
            e.g. [[0,1], [8]] on MNIST would be a binary classification problem where the first class
            consists of samples of 0's and 1's and the second class has samples of 8's
        onehot (boolean): whether to use one-hot label encodings (typical for MSE loss). Default: True
        format (str): specify order of (sample, channel, height, width) dims. 'NCHW' default, 'N' (other
            axes having been reshaped), or 'NHWC.'
            torchvision.dataset('cifar10') uses latter, needs ToTensor transform to reshape; former is ready-to-use.
        """

        assert dataset_name in self.dataset_dict
        self.name = dataset_name

        def format_data(dataset):
            if self.name in ['cifar10','cifar100']:
                X, y = dataset.data, dataset.targets
                X = X.transpose(0, 3, 1, 2)
                y = np.array(y)
            if self.name in ['mnist', 'fmnist']:
                X, y = dataset.data.numpy(), dataset.targets.numpy()
                X = X[:, None, :,:]
            if self.name in ['svhn']:
                X, y = dataset.data, dataset.labels
            if self.name in ['imagenet32', 'imagenet64']:
                X, y = dataset['data'], dataset['labels']
                X = X.reshape(-1, 3, 32, 32)
                y -= 1
            assert format in ['NHWC', 'N', 'NCHW']
            if format == 'NHWC':
                X = X.transpose(0, 2, 3, 1)
            elif format == 'N':
                X = X.reshape(X.shape[0], -1)

            if classes is not None:
                # convert old class labels to new
                converter = -1 * np.ones(int(max(y)) + 1)
                for new_class, group in enumerate(classes):
                    group = [group] if type(group) == int else group
                    for old_class in group:
                        converter[old_class] = new_class
                # remove datapoints not in new classes
                mask = (converter[y] >= 0)
                X = X[mask]
                y = converter[y][mask]

            # make elements of input O(1)
            X = X/255.0
            # shape labels (N, nclasses)
            y = F.one_hot(torch.Tensor(y).long()).numpy() if onehot else y[:, None]

            return X.astype(np.float32), y.astype(np.float32)

        if self.name in ['cifar10','cifar100', 'mnist', 'fmnist']:
            raw_train = self.dataset_dict[self.name](root=data_dir, train=True, download=True, transform=kwargs.get("transform", None))
            raw_test = self.dataset_dict[self.name](root=data_dir, train=False, download=True, transform=kwargs.get("transform", None))
        if self.name == 'svhn':
            raw_train = self.dataset_dict[self.name](root=data_dir, split='train', download=True)
            raw_test = self.dataset_dict[self.name](root=data_dir, split='test', download=True)
        if self.name in ['imagenet32', 'imagenet64']:
            raw_train = np.load(f"{data_dir}/{self.name}-val.npz")
            raw_test = np.load(f"{data_dir}/{self.name}-val.npz")

        # process raw datasets
        self.train_X, self.train_y = format_data(raw_train)
        self.test_X, self.test_y = format_data(raw_test)

    def get_dataset(self, n, get="train", rng=None, binarize=False, centered=False, normalize=False, **datasetargs):
        """Generate an image dataset.

        n (int): the dataset size
        rng (numpy RNG): numpy RNG state for random sampling. Default: None
        get (str): either "train" or "test." Default: "train"
        binarize (bool): binarizes labels to +- 1. Default: False
        centered (bool): centers the data across samples. Default: False
        normalize (bool): normalizes the data, mean over samples, variance over pixels. Default: False

        Returns: tuple (X, y) such that X.shape = (n, *in_shape), y.shape = (n, *out_shape)
        """

        assert int(n) == n
        n = int(n)
        assert n > 0
        assert get in ["train", "test"]
        full_X, full_y = (self.train_X, self.train_y) if get == "train" else (self.test_X, self.test_y)

        def center_data(X:np.ndarray):
            return X - X.mean(axis=0)  

        if binarize:
            full_y = 2*full_y - 1

        if centered:
            full_X = center_data(full_X)

        if normalize:
            full_X = center_data(full_X)
            full_X /= np.linalg.norm(full_X, axis=tuple(range(1, full_X.ndim)), keepdims=True)
            
        # get subset
        idxs = slice(n) if rng is None else rng.choice(len(full_X), size=n, replace=False)
        X, y = full_X[idxs].copy(), full_y[idxs].copy()
        assert len(X) == n
        return X, y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label