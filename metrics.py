import numpy as np
from skimage.metrics import structural_similarity as ssim

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim_metric(img1, img2):
    return ssim(img1, img2, channel_axis=-1)