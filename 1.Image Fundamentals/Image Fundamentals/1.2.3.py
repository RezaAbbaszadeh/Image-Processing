import cv2
import numpy as np


def quantize(img, bit):
    return np.floor(img / (2 ** (8 - bit))).astype(np.uint8) * (2 ** (8 - bit))


img = cv2.imread('Barbara.bmp', cv2.IMREAD_GRAYSCALE)
for i in range(1, 9):
    q = quantize(img, i)
    cv2.imwrite("res1.2.3-" + str(i) + ".jpg", q)
