import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt

from common import median_filter

img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE)
densities = [0.05, 0.1, 0.2]
window_sizes = [3, 5, 7, 9, 11]
MSEs = []
for density in densities:
    noisy_img = skimage.util.random_noise(img, mode='s&p', seed=None, clip=True, amount=density)
    noisy_img = skimage.img_as_ubyte(noisy_img)
    cv2.imwrite('3-2-1/3-2-1-den{}.jpg'.format(density), noisy_img)
    MSEs_of_density = ['p = {}'.format(density)]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0][0].imshow(noisy_img, cmap='gray', aspect='auto')
    axs[0][0].set_title('Noisy image')
    axs[0][0].axis('off')
    i = 0
    for size in window_sizes:
        res = median_filter(noisy_img, size)
        MSEs_of_density.append(np.square(np.subtract(img, res)).mean())

        i += 1
        axs[int(i / 3)][i % 3].imshow(res, cmap='gray', aspect='auto')
        axs[int(i / 3)][i % 3].set_title('{}x{}'.format(size, size))
        axs[int(i / 3)][i % 3].axis('off')
    fig.savefig('3-2-1/3-2-1-den{}-res.jpg'.format(density))
    MSEs.append(MSEs_of_density)

col_labels = ("       ", "3X3", "5x5", "7x7", "9x9", "11x11")
fig, ax = plt.subplots(dpi=300, figsize=(5, 1))
ax.axis('off')
ax.table(cellText=MSEs, colLabels=col_labels, loc='center')
fig.savefig('3-2-1/3-2-1-resTable.png')
