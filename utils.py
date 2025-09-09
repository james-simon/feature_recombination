import numpy as np
import torch


def ensure_numpy(x, dtype=np.float64, clone=False):
    """Convert torch.Tensor to numpy array if necessary.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    result = np.asarray(x, dtype=dtype)
    if clone and result is x:
        return result.copy()
    return result


def ensure_torch(x, dtype=torch.float32, device=None, clone=False):
    """
    Convert array-like to torch.Tensor on the desired device/dtype.

    Notes:
      - In multiprocessing, call torch.cuda.set_device(device_id) in each worker.
        Leaving device=None will then use that worker's GPU via 'cuda'.
      - On single-GPU, device=None -> 'cuda' automatically.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")  # current CUDA device (per-process)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)

    if not isinstance(x, torch.Tensor):
        return torch.as_tensor(x, dtype=dtype, device=device)

    # Already a tensor: move/cast only if needed
    needs_move = (x.device != device) or (x.dtype != dtype)
    if needs_move:
        x = x.to(device=device, dtype=dtype, non_blocking=True)

    return x.clone() if clone else x