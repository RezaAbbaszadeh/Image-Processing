import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt

from common import normalize, psnr


def quantize(arr, value=2):
    return (np.int64(arr) / value) * value


img = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
C = pywt.wavedec2(img, 'haar', 'periodization', 3)

plt.figure(figsize=(30, 30))

cA3 = C[0]
(cH1, cV1, cD1) = C[-1]
(cH2, cV2, cD2) = C[-2]
(cH3, cV3, cD3) = C[-3]

cA3_quantized = quantize(cA3)
cH3_quantized = quantize(cH3)
cV3_quantized = quantize(cV3)
cD3_quantized = quantize(cD3)
cH2_quantized = quantize(cH2)
cV2_quantized = quantize(cV2)
cD2_quantized = quantize(cD2)
cH1_quantized = quantize(cH1)
cV1_quantized = quantize(cV1)
cD1_quantized = quantize(cD1)
C_quantized = [cA3,
               (cH3_quantized, cV3_quantized, cD3_quantized),
               (cH2_quantized, cV2_quantized, cD2_quantized),
               (cH1_quantized, cV1_quantized, cD1_quantized)]

arr, coeff_slices = pywt.coeffs_to_array(C_quantized)
plt.figure()
plt.imshow(arr, cmap=plt.cm.gray)
cv2.imwrite('6-1-6/pyramid.jpg', normalize(arr))

imgr_quantized = pywt.waverec2(C_quantized, 'haar', 'periodization')
imgr_quantized = np.uint8(imgr_quantized)
plt.figure()
plt.imshow(imgr_quantized, cmap=plt.cm.gray)
plt.title('reconstructed quantized', fontsize=20)
plt.savefig('6-1-6/reconstructed quantized.jpg')

psnr, mse = psnr(img, imgr_quantized)
print('psnr: {}'.format(psnr))
print('mse: {}'.format(mse))
