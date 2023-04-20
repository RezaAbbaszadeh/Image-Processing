from math import floor

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import rect_to_polar, log_ft, polar_to_rect

img = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
m = img.shape[0]
n = img.shape[1]
p = 2 * m
q = 2 * n
fp = np.zeros((p, q))
fp[0:m, 0:n] = img
img_ft = np.fft.fft2(fp)
img_amp, img_phase = rect_to_polar(img_ft)
img_amp = np.fft.fftshift(img_amp)
cv2.imwrite('4.1.1/Lena-FT.png', log_ft(img_amp))

a = np.asarray([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])
a_row = np.asarray([[1, 2, 1]])
a_col = np.asarray([[1], [2], [1]])
b = np.asarray([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])
c = np.asarray([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

filters = [a_row, a_col, a, b, c]
filters_coe = [1 / 4, 1 / 4, 1 / 16, 1, 1]
filter_names = ['A-row', 'A-col', 'A', 'B', 'C']
for idx, filter in enumerate(filters):
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    padded_filter = np.zeros((p, q))
    padded_filter[
    floor(p / 2) - floor(filter.shape[0] / 2):floor(p / 2) + floor(filter.shape[0] / 2 + 1),
    floor(q / 2) - floor(filter.shape[1] / 2):floor(q / 2) + floor(filter.shape[1] / 2 + 1)
    ] = filters_coe[idx] * filter

    H = np.fft.fft2(padded_filter)
    H = np.fft.fftshift(H)
    HA, HP = rect_to_polar(H)

    axs[0].imshow(log_ft(HA), cmap='gray', aspect='auto')
    axs[0].set_title('filter {} in frequency domain'.format(filter_names[idx]))
    axs[0].axis('off')

    res_ft = np.multiply(img_amp, HA)

    axs[1].imshow(log_ft(res_ft), cmap='gray', aspect='auto')
    axs[1].set_title('image after filter in frequency domain')
    axs[1].axis('off')

    res_ft = np.fft.ifftshift(res_ft)
    res_ft = polar_to_rect(res_ft, img_phase)
    res = np.fft.ifft2(res_ft)
    res = np.real(res[0:m, 0:n])

    axs[2].imshow(res, cmap='gray', aspect='auto')
    axs[2].set_title('final result')
    axs[2].axis('off')

    fig.savefig('4.1.1/{}.jpg'.format(filter_names[idx]), bbox_inches='tight')
