import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt

from common import normalize, show_wt, psnr

img = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
C = pywt.wavedec2(img, 'haar', 'periodization', 3)

plt.figure(figsize=(30, 30))

cA3 = C[0]
(cH1, cV1, cD1) = C[-1]
(cH2, cV2, cD2) = C[-2]
(cH3, cV3, cD3) = C[-3]
show_wt(np.zeros((0, 0)), cH1, cV1, cD1, '6-1-5/Level 1.jpg')
show_wt(np.zeros((0, 0)), cH2, cV2, cD2, '6-1-5/Level 2.jpg')
show_wt(C[0], cH3, cV3, cD3, '6-1-5/Level 3.jpg')

arr, coeff_slices = pywt.coeffs_to_array(C)
plt.figure()
plt.imshow(arr, cmap=plt.cm.gray)
cv2.imwrite('6-1-5/pyramid.jpg', normalize(arr))

imgr = pywt.waverec2(C, 'haar', 'periodization')
imgr = np.uint8(imgr)
plt.figure()
plt.imshow(imgr, cmap=plt.cm.gray)
plt.title('reconstructed', fontsize=20)
plt.savefig('6-1-5/reconstructed.jpg')

psnr, mse = psnr(img, imgr)
print('psnr: {}'.format(psnr))
print('mse: {}'.format(mse))

