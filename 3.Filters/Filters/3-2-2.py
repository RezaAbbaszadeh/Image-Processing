import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt

from common import median_filter, box_filter

img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE)
variances = [0.05, 0.1, 0.2]
sizeWndws = [3, 5, 7, 9, 11]
MSEs_median = []
MSEs_box = []
for variance in variances:
    noisy_img = skimage.util.random_noise(img, mode='gaussian', seed=None, clip=True, var=variance)
    noisy_img = skimage.img_as_ubyte(noisy_img)
    cv2.imwrite('3-2-2/noisy-den{}.jpg'.format(variance), noisy_img)
    MSEs_of_density_median = ['p = {}'.format(variance)]
    MSEs_of_density_box = ['p = {}'.format(variance)]

    fig_box, axs_box = plt.subplots(2, 3, figsize=(15, 10))
    axs_box[0][0].imshow(noisy_img, cmap='gray', aspect='auto')
    axs_box[0][0].set_title('Noisy image')
    axs_box[0][0].axis('off')

    fig_median, axs_median = plt.subplots(2, 3, figsize=(15, 10))
    axs_median[0][0].imshow(noisy_img, cmap='gray', aspect='auto')
    axs_median[0][0].set_title('Noisy image')
    axs_median[0][0].axis('off')
    i = 0
    for size in sizeWndws:
        i += 1
        median = median_filter(noisy_img, size)
        MSEs_of_density_median.append(np.square(np.subtract(img, median)).mean())
        axs_median[int(i / 3)][i % 3].imshow(median, cmap='gray', aspect='auto')
        axs_median[int(i / 3)][i % 3].set_title('{}x{}'.format(size, size))
        axs_median[int(i / 3)][i % 3].axis('off')

        box = box_filter(noisy_img, size)
        MSEs_of_density_box.append(np.square(np.subtract(img, box)).mean())
        axs_box[int(i / 3)][i % 3].imshow(box, cmap='gray', aspect='auto')
        axs_box[int(i / 3)][i % 3].set_title('{}x{}'.format(size, size))
        axs_box[int(i / 3)][i % 3].axis('off')

    fig_median.savefig('3-2-2/den{}-median.jpg'.format(variance))
    fig_box.savefig('3-2-2/den{}-box.jpg'.format(variance))

    MSEs_median.append(MSEs_of_density_median)
    MSEs_box.append(MSEs_of_density_box)


col_labels = ("       ", "3X3", "5x5", "7x7", "9x9", "11x11")
fig, ax = plt.subplots(dpi=300, figsize=(5, 1))
ax.axis('off')
ax.table(cellText=MSEs_median, colLabels=col_labels, loc='center')
fig.savefig('3-2-2/table_median.png')

col_labels = ("       ", "3X3", "5x5", "7x7", "9x9", "11x11")
fig, ax = plt.subplots(dpi=300, figsize=(5, 1))
ax.axis('off')
ax.table(cellText=MSEs_box, colLabels=col_labels, loc='center')
fig.savefig('3-2-2/table_box.png')