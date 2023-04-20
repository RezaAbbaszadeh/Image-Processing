import cv2
import matplotlib.pyplot as plt
import numpy as np

from common import approximation_pyramid, laplacian_pyramid, reconstruct_from_laplacian, psnr, normalize

img = cv2.imread("mona lisa.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite('6-1-1/img.png', img)
img = img.astype(float)
gaussian_pyramid = approximation_pyramid(img, 5)
gauss_pyramid_joined = np.ones((img.shape[0], 2 * img.shape[1])) * 255
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
last_pixel = 0
for i, image in enumerate(gaussian_pyramid):
    axes[i // 3, i % 3].imshow(image, cmap='gray')
    gauss_pyramid_joined[img.shape[0] - image.shape[0]:img.shape[0], last_pixel:last_pixel + image.shape[1]] = normalize(image)
    last_pixel += image.shape[1]

fig.savefig('6-1-1/approximation_pyramid.png')
cv2.imwrite('6-1-1/approximation_pyramid_joined.jpg', gauss_pyramid_joined)

laplacian_pyramid = laplacian_pyramid(gaussian_pyramid)
laplacian_pyramid_joined = np.ones((img.shape[0], 2 * img.shape[1])) * 255

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
last_pixel = 0
for i, image in enumerate(laplacian_pyramid):
    laplacian_pyramid_joined[img.shape[0] - image.shape[0]:img.shape[0], last_pixel:last_pixel + image.shape[1]] = normalize(image)
    axes[i // 3, i % 3].imshow(image, cmap='gray')
    last_pixel += image.shape[1]

fig.savefig('6-1-1/laplace_pyramid.png')
cv2.imwrite('6-1-1/laplace_pyramid_joined.jpg', laplacian_pyramid_joined)
