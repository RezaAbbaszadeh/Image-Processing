import numpy as np
import cv2
import copy
import skimage

from common import box_filter, laplacian_filter, normalize

img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE)
box_filtered = copy.deepcopy(img)

# 3-1-3
repeats = 10
for i in range(0, repeats):
    box_filtered = box_filter(box_filtered, window_size=3)
    cv2.imwrite('3-1/box-filtered-3x3-it{}.jpg'.format(i + 1), box_filtered)

# 3-1-4
window_sizes = [3, 9, 15, 27]
noisy_img = skimage.util.random_noise(img, mode='s&p', seed=None, clip=True, amount=0.05)
noisy_img = skimage.img_as_ubyte(noisy_img)
cv2.imwrite('3-1/noise.jpg', noisy_img)
for size in window_sizes:
    res = box_filter(noisy_img, window_size=size)
    cv2.imwrite('3-1/box-filtered-{}x{}.jpg'.format(size, size), res)

# 3-1-5
filtered_noisy = box_filter(noisy_img, 5)
cv2.imwrite('3-1/noise-reduction.jpg', filtered_noisy)


# 3-1-6
sharpened = copy.deepcopy(img)
for i in range(0, 3):
    filtered = laplacian_filter(sharpened, np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    filtered_norm = normalize(filtered)
    sharpened = normalize(filtered_norm + img)
    cv2.imwrite('3-1/laplacian-filtered-{}.jpg'.format(i), sharpened)
