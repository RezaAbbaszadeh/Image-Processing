import cv2
import matplotlib.pyplot as plt
import numpy as np


def down_sample_simple(img):
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] / 2)
    res = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            res[i, j] = img[i * 2, j * 2]
    return res


def down_sample_average(img):
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] / 2)
    res = np.zeros((height, width))
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            res[i, j] = (int(img[i * 2, j * 2]) + int(img[i * 2, j * 2 + 1]) +
                         int(img[i * 2 + 1, j * 2]) + int(img[i * 2 + 1, j * 2 + 1])) / 4
    return res


def up_sample_pixel_replication(img):
    width = img.shape[1] * 2
    height = img.shape[0] * 2
    res = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            res[i, j] = img[int(i / 2), int(j / 2)]
    return res


def up_sample_bilinear_interpolation(img):
    width = img.shape[1] * 2
    height = img.shape[0] * 2
    res = np.zeros((height, width))
    for i in range(0, height):
        if i % 2 == 0:
            for j in range(0, width):
                if j % 2 == 0:
                    res[i, j] = img[int(i / 2), int(j / 2)]

    for i in range(1, height-2, 2):
        for j in range(1, width-2, 2):
            x_y = np.array([[i + 1, j - 1, (i + 1) * (j - 1), 1],
                            [i + 1, j + 1, (i + 1) * (j + 1), 1],
                            [i - 1, j + 1, (i - 1) * (j + 1), 1],
                            [i - 1, j - 1, (i - 1) * (j - 1), 1]])
            v = np.array(
                [res[i + 1, j - 1], res[i + 1, j + 1], res[i - 1, j + 1],
                 res[i - 1, j - 1]])
            x = np.linalg.solve(x_y, v)
            res[i, j] = x[0] * i + x[1] * j + x[2] * i * j + x[3]

    for i in range(1, height - 2):
        if i % 2 == 1:
            k = 2
        else:
            k = 1
        for j in range(k, width - 2, 2):
            x_y = np.array([[i + 1, j, (i + 1) * (j), 1],
                            [i, j + 1, (i) * (j + 1), 1],
                            [i - 1, j, (i - 1) * (j), 1],
                            [i, j - 1, (i) * (j - 1), 1]])

            v = np.array([res[i + 1, j], res[i, j + 1], res[i - 1, j],
                          res[i, j - 1]])
            if res[i, j] == 0:
                try:
                    x = np.linalg.solve(x_y, v)
                    res[i, j] = x[0] * (i) + x[1] * j + x[2] * i * j + x[3]
                except:
                    res[i, j] = (res[i + 1, j] + res[i, j + 1] + res[i - 1, j] +
                                  res[i, j - 1]) / 4

    return res


img = cv2.imread('Goldhill.bmp', cv2.IMREAD_GRAYSCALE)
down_avg = down_sample_average(img)
cv2.imwrite('down_avg.jpg', down_avg)
down_simple = down_sample_simple(img)
cv2.imwrite('down_simple.jpg', down_simple)

up_rep_down_avg = up_sample_pixel_replication(down_avg)
cv2.imwrite('up_rep_down_avg.jpg', up_rep_down_avg)
up_bilinear_down_avg = up_sample_bilinear_interpolation(down_avg)
cv2.imwrite('up_bilinear_down_avg.jpg', up_bilinear_down_avg)
up_rep_down_simple = up_sample_pixel_replication(down_simple)
cv2.imwrite('up_rep_down_simple.jpg', up_rep_down_simple)
up_bilinear_down_simple = up_sample_bilinear_interpolation(down_simple)
cv2.imwrite('up_bilinear_down_simple.jpg', up_bilinear_down_simple)

mse_up_rep_down_avg = np.square(np.subtract(img, up_rep_down_avg)).mean()
mse_up_bilinear_down_avg = np.square(np.subtract(img, up_bilinear_down_avg)).mean()
mse_up_rep_down_simple = np.square(np.subtract(img, up_rep_down_simple)).mean()
mse_up_bilinear_down_simple = np.square(np.subtract(img, up_bilinear_down_simple)).mean()
data = [["Average down", mse_up_rep_down_avg, mse_up_bilinear_down_avg],
        ["Simple down", mse_up_rep_down_simple, mse_up_bilinear_down_simple]]
col_labels = ("*", "Pixel replication up", "bilinear up")
fig, ax = plt.subplots(dpi=300, figsize=(5, 1))
ax.axis('off')
ax.table(cellText=data, colLabels=col_labels, loc='center')
fig.savefig('table1.2.2.png')
