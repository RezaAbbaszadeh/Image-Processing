import numpy as np
import cv2
from scipy import interpolate
from utils import *

img1 = cv2.imread('Car1.jpg')
img2 = cv2.imread('Car2.jpg')

point1_src = [835, 397]  # billboard
point1_dst = [415, 414]
point2_src = [438, 482]  # road beginning
point2_dst = [3, 505]
point3_src = [692, 703]  # grass
point3_dst = [273, 726]
point4_src = [440, 252]  # cloud
point4_dst = [4, 255]


v_w = np.array([[1, point1_dst[0], point1_dst[1], point1_dst[0] * point1_dst[1]],
                [1, point2_dst[0], point2_dst[1], point2_dst[0] * point2_dst[1]],
                [1, point3_dst[0], point3_dst[1], point3_dst[0] * point3_dst[1]],
                [1, point4_dst[0], point4_dst[1], point4_dst[0] * point4_dst[1]]])

x = np.array([point1_src[0], point2_src[0], point3_src[0], point4_src[0]])
c1, c2, c3, c4 = np.linalg.solve(v_w, x)

y = np.array([point1_src[1], point2_src[1], point3_src[1], point4_src[1]])
c5, c6, c7, c8 = np.linalg.solve(v_w, y)

bottom_left_y = int(c5 + (c6 * 0) + (c7 * img2.shape[0]) + (c8 * img2.shape[0] * 0))
bottom_left_x = int(c1 + (c2 * 0) + (c3 * img2.shape[0]) + (c4 * img2.shape[0] * 0))

bottom_right_y = int(c5 + (c6 * img2.shape[1]) + (c7 * img2.shape[0]) + (c8 * img2.shape[0] * img2.shape[1]))
bottom_right_x = int(c1 + (c2 * img2.shape[1]) + (c3 * img2.shape[0]) + (c4 * img2.shape[0] * img2.shape[1]))

top_right_y = int(c5 + (c6 * img2.shape[1]) + (c7 * 0) + (c8 * 0 * img2.shape[1]))
top_right_x = int(c1 + (c2 * img2.shape[1]) + (c3 * 0) + (c4 * 0 * img2.shape[1]))

top_left_y = int(c5)
top_left_x = int(c1)

min_x = min(top_left_x, bottom_left_x)
max_x = max(top_right_x, bottom_right_x)
min_y = min(top_right_y, top_left_y)
max_y = max(bottom_left_y, bottom_right_y)

img2_converted = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=int)

for i in range(0, img2.shape[0]):
    for j in range(0, img2.shape[1]):
        for k in range(3):
            xnew = int(c1 + (c2 * j) + (c3 * i) + (c4 * i * j)) - min_x
            ynew = int(c5 + (c6 * j) + (c7 * i) + (c8 * i * j)) - min_y
            img2_converted[ynew, xnew, k] = img2[i, j, k]

cv2.imwrite("img2-converted.jpg", img2_converted)
img2_converted = nearest_interpolate(img2_converted, max_distance=2)
cv2.imwrite("img2-interpolated.jpg", img2_converted)
min_y -= 2
max_y += 2
min_x -= 2
max_x += 2

stitched_img = np.zeros((max_y - min(0, min_y), max_x, 3), dtype=int)

for i in range(0, img1.shape[0]):
    for j in range(0, img1.shape[1]):
        for k in range(3):
            stitched_img[i - min(0, min_y), j, k] = img1[i, j, k]

for i in range(0, max_y - min_y):
    for j in range(0, max_x - min_x):
        for k in range(3):
            if img2_converted[i, j, k] != 0:
                stitched_img[i - min(0, min_y) + min_y, j + min_x, k] = img2_converted[i, j, k]

cv2.imwrite("res-1.1.2.jpg", stitched_img)
