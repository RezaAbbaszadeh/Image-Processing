import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt

from common import compute_histogram, localHistEq, globalHistEq, normalize

img = cv2.imread('Camera Man.bmp', cv2.IMREAD_GRAYSCALE)
histogram = compute_histogram(img)
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
axs[0].imshow(img, cmap='gray', aspect='equal')
axs[1].bar(range(0, 256), histogram)
fig.savefig('2.1.1.jpg')

# 2.1.1.1
D = (copy.deepcopy(img) / 3).astype(np.uint8)
cv2.imwrite('2.1.1.1-D.jpg', D)

# 2.1.1.2
histogramD = compute_histogram(D)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0][0].imshow(img, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1][0].bar(range(0, 256), histogram)
axs[0][1].imshow(D, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1][1].bar(range(0, 256), histogramD)
fig.savefig('2.1.1.2.jpg')

# 2.1.1.3
H = globalHistEq(D)
cv2.imwrite('2.1.1.3-H.jpg', H)

# 2.1.1.4
L = localHistEq(D, 100, 30)
cv2.imwrite('2.1.1.4-L.jpg', L)

# 2.1.1.5
histH = compute_histogram(H)
histL = compute_histogram(L)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0][0].imshow(H, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1][0].bar(range(0, 256), histH)
axs[0][1].imshow(L, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1][1].bar(range(0, 256), histL)
fig.savefig('2.1.1.5.jpg')

# 2.1.1.6
log = D.astype(float)
inv_log = D.astype(int)
power = D.astype(float)
root = D.astype(float)

log = normalize(np.log2(log, out=np.zeros_like(log), where=(log != 0)))
inv_log = normalize(np.exp(inv_log))
power = normalize(power ** 2)
root = normalize(np.sqrt(root))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(log, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1].bar(range(0, 256), compute_histogram(log))
fig.savefig('2.1.1.6-log.jpg')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(inv_log, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1].bar(range(0, 256), compute_histogram(inv_log))
fig.savefig('2.1.1.6-inv-log.jpg')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(power, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1].bar(range(0, 256), compute_histogram(power))
fig.savefig('2.1.1.6-power.jpg')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(root, cmap='gray', vmin=0, vmax=255, aspect='equal')
axs[1].bar(range(0, 256), compute_histogram(root))
fig.savefig('2.1.1.6-root.jpg')
