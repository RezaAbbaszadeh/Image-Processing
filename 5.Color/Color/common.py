import numpy as np


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min) * 255
    return img


def compute_psnr(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    if mse == 0:
        return np.inf, 0
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr, mse
