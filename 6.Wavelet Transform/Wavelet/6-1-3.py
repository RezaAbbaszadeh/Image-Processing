import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import approximation_pyramid, laplacian_pyramid, reconstruct_from_laplacian, psnr, normalize

img = cv2.imread("Lena.bmp", cv2.IMREAD_GRAYSCALE)
cv2.imwrite('6-1-3/img.png', img)
img = img.astype(float)
gaussian_pyramid = approximation_pyramid(img, 8)
gauss_pyramid_joined = np.ones((img.shape[0], 2 * img.shape[1])) * 255
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
last_pixel = 0
for i, image in enumerate(gaussian_pyramid):
    axes[i // 5, i % 5].imshow(image, cmap='gray')
    gauss_pyramid_joined[img.shape[0] - image.shape[0]:img.shape[0], last_pixel:last_pixel + image.shape[1]] = normalize(image)
    last_pixel += image.shape[0]

fig.savefig('6-1-3/approximation_pyramid.png')
cv2.imwrite('6-1-3/approximation_pyramid_joined.jpg', gauss_pyramid_joined)

laplacian_pyramid = laplacian_pyramid(gaussian_pyramid)
laplacian_pyramid_joined = np.ones((img.shape[0], 2 * img.shape[1])) * 255

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
last_pixel = 0
for i, image in enumerate(laplacian_pyramid):
    laplacian_pyramid_joined[img.shape[0] - image.shape[0]:img.shape[0], last_pixel:last_pixel + image.shape[1]] = normalize(image)
    axes[i // 5, i % 5].imshow(image, cmap='gray')
    last_pixel += image.shape[0]

fig.savefig('6-1-3/laplace_pyramid.png')
cv2.imwrite('6-1-3/laplace_pyramid_joined.jpg', laplacian_pyramid_joined)
reconstruct = reconstruct_from_laplacian(laplacian_pyramid, gaussian_pyramid[len(gaussian_pyramid) - 1])
cv2.imwrite('6-1-3/reconstruct.png', reconstruct)
psnr, mse = psnr(img, reconstruct)
print("mmse={}".format(mse))
print("psnr={}".format(psnr))
