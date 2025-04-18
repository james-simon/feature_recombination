import numpy as np
import torch as torch
import torchvision
import torch.nn.functional as F

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

    def __init__(self, dataset_name, data_dir, classes=None, onehot=True, format='NCHW'):
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
            raw_train = self.dataset_dict[self.name](root=data_dir, train=True, download=True)
            raw_test = self.dataset_dict[self.name](root=data_dir, train=False, download=True)
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

    # def _get_imagenet(self, class_url_dict):
    #     # from torch.utils.data import Dataset, DataLoader
    #     from PIL import Image
    #     from io import BytesIO
    #     # import torchvision.transforms as transforms

    #     self.data = []
    #     self.labels = []
    #     self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_url_dict.keys())}
        
    #     for cls_name, urls in class_url_dict.items():
    #         label = self.class_to_idx[cls_name]
    #         count = 0
    #         for url in urls:
    #             try:
    #                 response = requests.get(url, timeout=5)
    #                 img = Image.open(BytesIO(response.content)).convert("RGB")
    #                 self.data.append(img)
    #                 self.labels.append(label)
    #                 count += 1
    #                 if count >= max_per_class:
    #                     break
    #             except Exception as e:
    #                 print(f"Skipping broken image: {url} ({e})")
    
    # def _get_imagenet_urls(synset_dict, max_per_class=10):
    #     import requests
    #     class_url_dict = {}
    #     for class_name, synset in synset_dict.items():
    #         try:
    #             url = f"https://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset}"
    #             response = requests.get(url)
    #             urls = response.text.splitlines()
    #             class_url_dict[class_name] = urls[:max_per_class]
    #         except Exception as e:
    #             print(f"Failed to fetch URLs for {class_name}: {e}")
    #             class_url_dict[class_name] = []
    #     return class_url_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# def test_imagenet(class_synsets, max_per_class=100):
#     import requests

#     def get_imageset(synset='n01440764'):    
#         url = f"https://www.image-net.org/api/text/imagenet.synset.geturls?wnid={synset}"
#         response = requests.get(url)
#         image_urls = response.text.splitlines()

#     class_dict = {synset: get_imageset(synset) for synset in class_synsets}
