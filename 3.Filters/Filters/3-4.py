import numpy as np
import cv2

from common import normalize


def gradient(img, window):
    res = np.zeros(img.shape)
    padding_v = int(window.shape[0] / 2)
    padding_h = int(window.shape[1] / 2)
    is_odd = 1
    if window.shape[0] == 2:
        is_odd = 0

    for i in range(padding_v, img.shape[0] - padding_v):
        for j in range(padding_h, img.shape[1] - padding_h):
            res[i, j] = abs(sum(np.asarray(img[i - padding_v:i + padding_v + is_odd,
                                           j - padding_h:j + padding_h + is_odd] * window).flatten()))

    return res


a = (1 / 2) * np.array([
    [1, 0, -1]
])

b = (1 / 6) * np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

c = (1 / 8) * np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

d = np.array([
    [1, 0],
    [0, -1]
])

e = np.array([
    [0, 1],
    [-1, 0]
])

filter_names = ['a', 'b', 'c', 'd', 'e']
filters = [
    a, b, c, d, e
]

img = cv2.imread('Elaine.bmp', cv2.IMREAD_GRAYSCALE)

for i in range(len(filters)):
    edge = normalize(gradient(img, filters[i]))
    cv2.imwrite('3-4-1/edge-{}.jpg'.format(filter_names[i]), edge)
    cv2.imwrite('3-4-1/res-{}.jpg'.format(filter_names[i]), img + edge)
