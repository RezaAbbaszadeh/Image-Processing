import numpy as np
import cv2
from skimage.color import rgb2gray
from scipy import signal as sig
import scipy.ndimage as ndi


def gradient_x(img):
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    return sig.convolve2d(img, kernel_x, mode='same')


def gradient_y(img):
    kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    return sig.convolve2d(img, kernel_y, mode='same')


img = cv2.imread('Building.jpg')
img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
gray_img = rgb2gray(img)
I_x = gradient_x(gray_img)
I_y = gradient_y(gray_img)
Ixx = ndi.gaussian_filter(I_x ** 2, sigma=1)
Ixy = ndi.gaussian_filter(I_y * I_x, sigma=1)
Iyy = ndi.gaussian_filter(I_y ** 2, sigma=1)
k = 0.05
detA = Ixx * Iyy - Ixy ** 2
traceA = Ixx + Iyy
harris_response = detA - k * traceA ** 2
img_with_corners = np.copy(img)

threshold_corners = 0.2
corners_count = 0
for rowindex, response in enumerate(harris_response):
    for col, r in enumerate(response):
        if r > threshold_corners:
            img_with_corners[rowindex, col] = [30, 30, 200]
            corners_count += 1

print("harris corner counts:", str(corners_count))
cv2.imwrite('7-2/harris-corner.png', img_with_corners)

img_with_corners = np.copy(img)
img_copy_for_edges = np.copy(img)
window_w = 3
window_h = 3
window_w_half = window_w // 2
window_h_half = window_h // 2
corners_count = 0
for rowindex in range(window_h_half, ((harris_response.shape[0]) - window_h_half)):
    for col in range(window_w_half, (harris_response.shape[1]) - window_w_half):
        tile_corner = harris_response[rowindex - window_h_half:rowindex + 1 + window_h_half,
                      col - window_w_half:col + 1 + window_w_half]
        max_in_row = np.max(tile_corner)
        r = harris_response[rowindex, col]
        if r > threshold_corners and r >= max_in_row:
            img_with_corners[rowindex, col] = [30, 30, 200]
            corners_count += 1
print("non suppression corner counts:", str(corners_count))
cv2.imwrite('7-2/non_max.png', img_with_corners)
