import cv2
import matplotlib.pyplot as plt
from common import compute_histogram, globalHistEq

img = cv2.imread('Camera Man.bmp', cv2.IMREAD_GRAYSCALE)

histogram = compute_histogram(img)

eq_img = globalHistEq(img)
eq_histogram = compute_histogram(eq_img)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0][0].imshow(img, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1][0].bar(range(0, 256), histogram)
axs[0][1].imshow(eq_img, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1][1].bar(range(0, 256), eq_histogram)
fig.savefig('2.1.2.jpg')
