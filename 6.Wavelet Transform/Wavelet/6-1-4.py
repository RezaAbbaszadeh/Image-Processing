import matplotlib.pyplot as plt
import cv2
from common import *

img = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
img = img.astype(float)
gaussian_pyramid = approximation_pyramid(img, 3, 'average')
gauss_pyramid_joined = np.ones((img.shape[0], 2 * img.shape[1])) * 255

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
last_pixel = 0
for i, im in enumerate(gaussian_pyramid):
    axes[i].imshow(im, cmap='gray')
    gauss_pyramid_joined[0:im.shape[0], last_pixel:last_pixel + im.shape[1]] = im
    last_pixel += im.shape[0]

fig.savefig('6-1-4/approximation_pyramid.png')
cv2.imwrite('6-1-4/approximation_pyramid_joined.jpg', gauss_pyramid_joined)

l_pyramid_im = np.ones((img.shape[0], 2 * img.shape[1])) * 255
laplace_pyramid = laplacian_pyramid(gaussian_pyramid)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
last_pixel = 0
for i, im in enumerate(laplace_pyramid):
    l_pyramid_im[0:im.shape[0], last_pixel:last_pixel + im.shape[1]] = normalize(im)
    axes[i].imshow(im, cmap='gray')
    last_pixel += im.shape[0]

fig.savefig('6-1-4/laplace_pyramid.png')
cv2.imwrite('6-1-4/laplace_pyramid_im.jpg', l_pyramid_im)
reconstructed = reconstruct_from_laplacian(laplace_pyramid, gaussian_pyramid[len(gaussian_pyramid) - 1])
cv2.imwrite('6-1-4/reconstruct_im.jpg', reconstructed)
psnr, mmse = psnr(img, reconstructed)
print("mmse={}".format(mmse))
print("psnr={}".format(psnr))
