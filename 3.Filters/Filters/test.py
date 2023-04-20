import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[1][0].imshow(img, cmap='gray', aspect='auto')
axs[1][0].axis('off')
axs[1][1].imshow(img, cmap='gray', aspect='auto')
axs[1][1].axis('off')
axs[0][1].imshow(img, cmap='gray', aspect='auto')
axs[0][1].axis('off')
axs[0][0].imshow(img, cmap='gray', aspect='auto')
axs[0][0].axis('off')
fig.savefig('test.jpg')
