from math import cos, sin
import numpy as np
import cv2
from utils import *


def rotate(img, degree):
    w = int(img.shape[1])
    h = int(img.shape[0])
    result_size = max(w, h) * 2
    rotated = np.zeros((result_size, result_size), dtype=int)

    radian = degree / 180 * np.pi

    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[0]):
            x = (j - img.shape[1] / 2) * cos(radian) - (i - img.shape[0] / 2) * sin(radian)
            y = (j - img.shape[1] / 2) * sin(radian) + (i - img.shape[0] / 2) * cos(radian)
            rotated[int(y + rotated.shape[0] / 2), int(x + rotated.shape[1] / 2)] = img[i, j]

    rotated = crop_gray_image(rotated)
    return rotated


img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE)

res45 = rotate(img, 45)
cv2.imwrite('1.1.3-45.jpg', res45)
cv2.imwrite('1.1.3-45-interpolated.jpg', nearest_interpolate(res45))

res100 = rotate(img, 100)
cv2.imwrite('1.1.3-100.jpg', res100)
cv2.imwrite('1.1.3-100-interpolated.jpg', nearest_interpolate(res100))

res670 = rotate(img, 670)
cv2.imwrite('1.1.3-670.jpg', res670)
cv2.imwrite('1.1.3-670-interpolated.jpg', nearest_interpolate(res670))
