import numpy as np
import scipy as sp
import torch

from itertools import combinations_with_replacement

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

def monomial_of_array(array, monomial):
  return np.prod([array[idx] ** monomial[idx] for idx in monomial])

def monomial_fn(X, exponents):
  n, d = X.shape

  fn_vals = np.ones(n)
  for idx in exponents:
    fn_vals *= X[:, idx] ** exponents[idx]

  return fn_vals

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

def monomial_string(monomial):
  if monomial == {}:
    return "$f(x) = 1$"

  monomial_str = "$f(x) = "
  for idx in monomial:
    monomial_str += "x_{"+str(idx)+"}"
    if monomial[idx] != 1:
      monomial_str += f"^{monomial[idx]}"
  monomial_str += "$"

  return monomial_str

def monomial_order(monomial):
  return sum(monomial.values())

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


