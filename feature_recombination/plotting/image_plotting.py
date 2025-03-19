import matplotlib.pyplot as plt
import numpy as np

def show_img(img, grayscale=True, vextent=None, ax=None):
    nchan = 1 if grayscale else 3
    cmap = 'gray' if grayscale else None
    vmin, vmax = vextent if vextent is not None else None, None
    assert len(img) % nchan == 0
    imgsz = np.sqrt(int(len(img) / nchan))
    assert imgsz == int(imgsz)
    imgsz = int(imgsz)
    img = img.reshape(nchan, imgsz, imgsz).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    if ax is None:
        plt.figure(figsize=(1.5, 1.5))
        plt.axis('off')
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.show()
        return
    ax.axis('off')
    return ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)