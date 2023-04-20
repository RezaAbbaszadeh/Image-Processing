import cv2
import numpy as np
import matplotlib.pyplot as plt

from common import compute_psnr


def quantize(img, level):
    d = 255 / level
    return np.rint(img / d) * d


img = cv2.imread('Lena.bmp')
levels = [64, 32, 16, 8]
PSNRs = ["psnr"]
MSEs = ["mse"]
for level in levels:
    quantized = quantize(img, level)
    cv2.imwrite('5-2-1/{}-level.png'.format(level), quantized)
    psnr, immse = compute_psnr(img, quantized)
    PSNRs.append(psnr)
    MSEs.append(immse)
data = [MSEs, PSNRs]
labels = (" ",) + tuple(levels)
fig, ax = plt.subplots(dpi=200, figsize=(6, 1))
ax.axis('off')
ax.table(colLabels=labels, cellText=data, loc='center')

fig.savefig('5-2-1/result.png')
