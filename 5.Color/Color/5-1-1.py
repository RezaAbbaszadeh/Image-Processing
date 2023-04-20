import copy

import cv2
import numpy as np

from common import normalize


def rgb_to_hsi(img):
    src = copy.deepcopy(img)
    src = src.astype(float)
    src += 1
    r = np.float32(src[:, :, 2])
    g = np.float32(src[:, :, 1])
    b = np.float32(src[:, :, 0])
    theta = np.degrees(
        np.arccos(
            np.divide(
                ((r - g) + (r - b)) / 2,
                np.sqrt(np.power(r - g, 2) + ((r - b) * (g - b))) + 1
            )
        )
    )
    h = theta
    s = np.zeros(h.shape, dtype=float)
    i = (r + g + b) / 3
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            s[y, x] = 1 - (3 / (r[y, x] + g[y, x] + b[y, x]) * min([r[y, x], g[y, x], b[y, x]]))
            if b[y, x] > g[y, x]:
                h[y, x] = 360 - theta[y, x]
    return h, s, i


img = cv2.imread('Lena.bmp')
h, s, i = rgb_to_hsi(img)

h = normalize(h)
s = normalize(s)
i = normalize(i)

cv2.imwrite('5-1-1/h.png', h)
cv2.imwrite('5-1-1/s.png', s)
cv2.imwrite('5-1-1/i.png', i)
