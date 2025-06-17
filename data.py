import numpy as np
import torch as torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from utils import ensure_torch

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

    def get_dataset(self, n, get="train", rng=None, binarize=False, centered=False, normalize=False):
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
    
    def get_train_test_dataset(self, n_train, n_test, **kwargs):
        X_train, y_train = self.get_dataset(n_train, get='train', centered=kwargs.get("center", False), normalize=kwargs.get("normalize", False))
        X_test, y_test = self.get_dataset(n_test, get='test', centered=kwargs.get("center", False), normalize=kwargs.get("normalize", False))
        X_train, y_train, X_test, y_test = [ensure_torch(t) for t in (X_train, y_train, X_test, y_test)]
        X_train = rearrange(X_train, 'Ntrain c h w -> Ntrain (c h w)')
        X_test = rearrange(X_test, 'Ntest c h w -> Ntest (c h w)')

        return X_train, y_train, X_test, y_test
