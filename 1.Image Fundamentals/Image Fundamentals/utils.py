import numpy as np
import copy


def crop_gray_image(img, tol=0):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_rgb_image(img, tol=0):
    img_sum_z = np.sum(img, axis=2, keepdims=False)
    mask = img_sum_z > tol
    non_blank_area = np.ix_(mask.any(1), mask.any(0))
    cropped = img[non_blank_area[0][0][0]:non_blank_area[0][-1][-1],
              non_blank_area[1][0][0]:non_blank_area[1][-1][-1], :]
    return cropped


def add_padding(img, padSize=3):
    padded = np.zeros((img.shape[0] + padSize * 2, img.shape[1] + padSize * 2, img.shape[2]))
    padded[padSize:img.shape[0] + padSize, padSize:img.shape[1] + padSize, :] = img
    return padded


def nearest_interpolate(img, max_distance=2):
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    padding_size = max_distance
    img = add_padding(img, padding_size * 2)
    copied_img = copy.deepcopy(img)
    img = img.astype(np.float32)
    for i in range(padding_size, img.shape[0] - padding_size):
        for j in range(padding_size, img.shape[1] - padding_size):
            for k in range(0, img.shape[2]):
                if img[i, j, k] == 0:
                    distance = 1
                    found = False
                    while distance <= max_distance and not found:
                        for i2 in range(i - distance, i + distance + 1):
                            for j2 in range(j - distance, j + distance + 1):
                                if img[i2, j2, k] != 0:
                                    copied_img[i, j, k] = img[i2, j2, k]
                                    found = True
                                    break
                            if found:
                                break
                        distance += 1
        print("interpolation row", i, "done")
    return copied_img
