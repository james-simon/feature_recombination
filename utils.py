import numpy as np
import scipy as sp
import torch

from itertools import combinations_with_replacement
from itertools import product

#TODO: implement __getitem__ slicing for start:stop:step
class ExptTrace():

    @classmethod
    def multi_init(cls, num_init, var_names):
        return [cls(var_names) for _ in range(num_init)]

    def __init__(self, var_names):
        assert "val" not in var_names, f"variable name 'val' disallowed"
        self.var_names = var_names
        self.vals = {}
        self.valshape = None

    def __setitem__(self, key, val):
        if self.valshape is None:
            self.valshape = np.shape(val)
        assert np.shape(val) == self.valshape, f"value shape {np.shape(val)} != expected {self.valshape}"
        key = tuple(key)
        assert len(key) == len(self.var_names), f"num keys {len(key)} != num vars {len(self.var_names)}"
        assert key not in self.vals, f"key {key} already exists. overwriting not supported"
        self.vals[key] = val

    def __getitem__(self, key):
        assert self.valshape is not None, "must add items before getting"
        if not isinstance(key, tuple):
            key = (key,)
        assert len(key) <= len(self.var_names), f"num keys {len(key)} > num vars {len(self.var_names)}"

        key_axes = []
        for idx, var_name in enumerate(self.var_names):
            if idx >= len(key):
                key_axes.append(self.get_axis(var_name))  #keyless extra axes are full axes
                continue

            key_i = key[idx]

            if isinstance(key_i, slice):
                # Extract indices based on slicing
                full_axis = self.get_axis(var_name)
                key_axes.append(full_axis[key_i])
            else:
                key_axes.append([key_i])

        shape = [len(key_idx_extent) for key_idx_extent in key_axes]

        if np.prod(shape) == 1: #handles single-lengthed axes
            key = tuple(k[0] for k in key_axes)
            assert key in self.vals, f"key {key} not found"
            return self.vals[key]

        vals = np.zeros(shape + list(self.valshape))

        idx_maps = [{val: i for i, val in enumerate(axis)} for axis in key_axes]

        for key in product(*key_axes):
            shape_idxs = tuple(idx_maps[dim][val] for dim, val in enumerate(key))
            assert key in self.vals, f"key {key} not found"
            vals[shape_idxs] = self.vals[key]

        return vals

    # old implementation w/o slicing
    # def __getitem__(self, key):
    #     assert self.valshape is not None, "must add items before getting"
    #     key = tuple(key)
    #     assert len(key) == len(self.var_names), f"num keys {len(key)} != num vars {len(self.var_names)}"
    #     key_axes = []
    #     for idx, var_name in enumerate(self.var_names):
    #         key_i = key[idx]
    #         key_idx_extent = [key_i]
    #         if isinstance(key_i, slice):
    #             slice_is_full = all([x==None for x in [key_i.start, key_i.stop, key_i.step]])
    #             assert slice_is_full, f"slice start/stop/step not supported ({var_name})"
    #             key_idx_extent = self.get_axis(var_name)
    #         key_axes.append(key_idx_extent)
    #     shape = [len(key_idx_extent) for key_idx_extent in key_axes]
    #     if np.prod(shape) == 1:
    #         assert key in self.vals, f"key {key} not found"
    #         return self.vals[key]
    #     vals = np.zeros(shape + list(self.valshape))

    #     idx_maps = []
    #     for axis in key_axes:
    #         idx_maps.append({val: i for i, val in enumerate(axis)})
    #     for key in product(*key_axes):
    #         shape_idxs = tuple(idx_maps[dim][val] for dim, val in enumerate(key))
    #         assert key in self.vals, f"key {key} not found"
    #         vals[shape_idxs] = self.vals[key]

    #     return vals

    def get_axis(self, var_name):
        assert var_name in self.var_names, f"var {var_name} not found"
        idx = self.var_names.index(var_name)
        key_idx_extent = set()
        for keys in self.vals.keys():
            key_idx_extent.add(keys[idx])
        return sorted(list(key_idx_extent))

    def get(self, **kwargs):
        key = self._get_key(_mode='get', **kwargs)
        return self[key]

    def set(self, **kwargs):
        assert "val" in kwargs, f"no val given"
        val = kwargs["val"]
        key = self._get_key(_mode='set', **kwargs)
        self[key] = val

    def is_written(self, **kwargs):
        key = self._get_key(_mode='set', **kwargs)
        return key in self.vals

    def _get_key(self, _mode='set', **kwargs):
        for var_name in self.var_names:
            if _mode == 'set':
                assert var_name in kwargs, f"must specify var {var_name}"
            elif _mode == 'get':
                if var_name not in kwargs:
                    kwargs[var_name] = slice(None, None, None)
            assert kwargs[var_name] is not None, f"var {var_name} cannot be None"
        key = tuple([kwargs[var_name] for var_name in self.var_names])
        return key

    def serialize(self):
        return {
            "var_names": self.var_names,
            "vals": self.vals,
            "valshape": self.valshape
        }

    @classmethod
    def deserialize(cls, data):
        obj = cls(data["var_names"])
        obj.vals = data["vals"]
        obj.valshape = data["valshape"]
        return obj

def ensure_numpy(x):
    """Convert torch.Tensor to numpy array if necessary."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def ensure_torch(x, dtype=torch.float64):
    """Convert numpy array to torch.Tensor if needed, and ensure correct dtype."""

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype).to(DEVICE)
    return x

def rms(x):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.mean(x ** 2))
    elif isinstance(x, torch.Tensor):
        return torch.sqrt(torch.mean(x ** 2))
    else:
        raise TypeError("Input must be a numpy array or a torch tensor.")

def cos_sim(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def hermite(k, x):
    """Compute the k-th probabilist's Hermite polynomial at x, normalized to have unit norm against the standard normal distribution."""

    return sp.special.hermite(k)(x / 2 ** .5) * 2 ** (-k / 2) / sp.special.factorial(k) ** .5

def hermite_product(X, exponents):
  n, d = X.shape

  fn_vals = np.ones(n)
  for idx in exponents:
    fn_vals *= hermite(exponents[idx], X[:, idx])

  return fn_vals

def shuffle_indices(*tensors):
    """
    Shuffle specified dimensions of multiple numpy arrays using the same random permutation.

    Args:
        *tensors: Tuples of (tensor, indices_to_shuffle). Each tensor is a numpy array,
                  and indices_to_shuffle is an int or a tuple/list of ints specifying
                  which dimensions to shuffle.

    Returns:
        Tuple of shuffled tensors, in the same order as the input.
    """
    # Collect all axes to shuffle and ensure they have the same size
    permutation_axis_size = None
    for tensor, indices in tensors:
        if not isinstance(tensor, np.ndarray):
            raise TypeError("All tensors must be numpy arrays.")
        # Ensure indices are in a tuple
        if isinstance(indices, int):
            indices = (indices,)
        elif not isinstance(indices, (tuple, list)):
            raise TypeError("Indices must be an int or a tuple/list of ints.")
        for axis in indices:
            if axis < 0 or axis >= tensor.ndim:
                raise ValueError(f"Axis {axis} is out of bounds for tensor with shape {tensor.shape}.")
            axis_size = tensor.shape[axis]
            if permutation_axis_size is None:
                permutation_axis_size = axis_size
            elif permutation_axis_size != axis_size:
                raise ValueError(
                    f"All axes to be shuffled must have the same size. "
                    f"Axis {axis} has size {axis_size}, expected {permutation_axis_size}."
                )

    # Generate a single random permutation
    permutation = np.random.permutation(permutation_axis_size)

    # Apply the permutation along the specified axes
    shuffled_tensors = []
    for tensor, indices in tensors:
        shuffled_tensor = tensor.copy()
        if isinstance(indices, int):
            indices = (indices,)
        for axis in indices:
            if tensor.shape[axis] != permutation_axis_size:
                raise ValueError(
                    f"Tensor's axis size {tensor.shape[axis]} does not match permutation size {permutation_axis_size}."
                )
            # Build index slices for advanced indexing
            idx = [slice(None)] * tensor.ndim
            idx[axis] = permutation
            shuffled_tensor = shuffled_tensor[tuple(idx)]
        shuffled_tensors.append(shuffled_tensor)

    return tuple(shuffled_tensors)

def element_products(arr, orders):
    """
    Generate all products of elements from `arr` for the specified list of `orders`.

    Parameters:
        arr (np.ndarray): Input array of elements.
        orders (list of int): List of orders for which to compute element products.

    Returns:
        np.ndarray: Concatenated array of all element products for the specified orders.
    """
    all_products = []
    for order in orders:
        if order == 0:
            all_products.append(np.array([1]))
        elif order >= 1:
            # Generate all combinations with replacement of the specified order
            products = [np.prod(comb) for comb in combinations_with_replacement(arr, order)]
            all_products.append(np.array(products))
        else:
            raise ValueError("Each order must be a natural number (0, 1, 2, ...).")

    # Concatenate all results into a single numpy array
    return np.sort(np.concatenate(all_products))[::-1]


