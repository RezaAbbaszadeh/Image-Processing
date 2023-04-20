import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import convolve_filter, normalize


def unsharp_masking(img, k, window):
    smoothed = convolve_filter(img, window)
    mask = img - smoothed
    res = img + k * mask
    return normalize(res)


windowList = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]
alphas = [.1, .3, .6, .8, 1]
img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE).astype(np.float)

for window in windowList:
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0][0].imshow(img, cmap='gray', aspect='auto')
    axs[0][0].set_title('Normal image')
    axs[0][0].axis('off')
    i = 0
    for alpha in alphas:
        res = unsharp_masking(img, alpha, np.ones(window))
        i += 1
        axs[int(i / 3)][i % 3].imshow(res, cmap='gray', aspect='auto')
        axs[int(i / 3)][i % 3].set_title('alpha = {}'.format(alpha))
        axs[int(i / 3)][i % 3].axis('off')

    fig.savefig('3-5/res{}.jpg'.format(window))
