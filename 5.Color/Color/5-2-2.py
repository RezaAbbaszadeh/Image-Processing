import cv2
import numpy as np

from common import compute_psnr


def quantize(img, levelB, levelG, levelR):
    dr = 255 / levelR
    dg = 255 / levelG
    db = 255 / levelB
    quantized_r = (np.rint(img[:, :, 2] / dr) * dr)
    quantized_g = (np.rint(img[:, :, 1] / dg) * dg)
    quantized_b = (np.rint(img[:, :, 0] / db) * db)
    res = np.zeros(img.shape)
    res[:, :, 0] = quantized_b
    res[:, :, 1] = quantized_g
    res[:, :, 2] = quantized_r
    return res


img = cv2.imread('Lena.bmp')
quantized = quantize(img, 4, 8, 8)
cv2.imwrite('5-2-2/result.jpg', quantized)
psnr, mse = compute_psnr(img, quantized)
print("psnr:", str(psnr))
print("mse:", str(mse))
