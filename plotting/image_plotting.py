import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.general import ensure_numpy

def show_img(img, grayscale=True, vextent=None, ax=None, normalize=True):
    nchan = 1 if grayscale else 3
    cmap = 'gray' if grayscale else None
    vmin, vmax = vextent if vextent is not None else None, None
    assert len(img) % nchan == 0
    imgsz = np.sqrt(int(len(img) / nchan))
    assert imgsz == int(imgsz)
    imgsz = int(imgsz)
    img = img.reshape(nchan, imgsz, imgsz).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img/img.max() if normalize else img
    if ax is None:
        plt.figure(figsize=(1.5, 1.5))
        plt.axis('off')
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.show()
        return
    ax.axis('off')
    return ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)

def see_monomial(X_train: np.ndarray):
    U, S, Vt = np.linalg.svd(X_train, full_matrices=False)
    vmin, vmax = Vt[:64,:].min().item(), Vt[:64,:].max().item()
    PC_ims = (Vt - vmin) / (vmax - vmin)
    return PC_ims.cpu().numpy()

def compare_heatmaps(X1, X2, idx1, idx2, plot_axes=None):
    col1 = ensure_numpy(X1[:, idx1])
    col2 = ensure_numpy(X1[:, idx2])
    gdata1 = ensure_numpy(X2[:, idx1])
    gdata2 = ensure_numpy(X2[:, idx2])

    heatmap_data_1, _, _ = np.histogram2d(col1, col2, bins=100)
    heatmap_data_2, _, _ = np.histogram2d(gdata1, gdata2, bins=100)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) if plot_axes is None else plot_axes
    
    sns.heatmap(heatmap_data_1.T, 
                xticklabels=False, 
                yticklabels=False, 
                cmap='viridis', ax=axes[0])
    axes[0].set_title("Normal data")

    sns.heatmap(heatmap_data_2.T, 
                xticklabels=False, 
                yticklabels=False, 
                cmap='viridis', ax=axes[1])
    axes[1].set_title("Gaussianized data")
    plt.tight_layout()
    plt.show()