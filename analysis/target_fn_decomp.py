import numpy as np
import torch
from utils import ensure_torch
from utils.stats import get_standard_tools

def get_vtilde(X_train, y_train, top_fra_eigmode=None, method = "LSTSQ", kerneltype="gaussian", kernel_width=1):
    assert method in ["LSTSQ", "dotprod"], "method for finding eigencoeffs not found"
    y_non_onehot = torch.argmax(y_train, dim=1) if len(y_train.shape) > 1 else y_train
    X_train /= torch.sqrt((torch.linalg.svdvals(X_train)**2).sum())

    monomials, kernel, H, fra_eigvals, data_eigvals = get_standard_tools(X_train, kerneltype, kernel_width, top_mode_idx=top_fra_eigmode)
    if method == "LSTSQ":
        v_tilde = torch.linalg.lstsq(ensure_torch(H), ensure_torch(y_non_onehot).unsqueeze(1)).solution.squeeze()
    elif method == "dotprod":
        v_tilde = (H.T @ y_non_onehot.float())
    return v_tilde, monomials

def get_y_recon(H, v_tilde, y, classes=10):
    y_pred = (ensure_torch(H) @ v_tilde).squeeze()
    # err = ((y-y_pred)**2).mean()
    y_non_onehot_np = y.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    y_pred_sorted = [y_pred_np[y_non_onehot_np == i] for i in range(len(classes))]
    return y_pred_sorted