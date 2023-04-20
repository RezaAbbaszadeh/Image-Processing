import cv2
import numpy as np

from common import globalHistEq, median_filter, convolve_filter, unsharp_masking, normalize, box_filter


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


sobel_v = (1 / 8) * np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

sobel_h = (1 / 8) * np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

for i in [5]:
    img = cv2.imread('noisy{}.jpg'.format(i), cv2.IMREAD_GRAYSCALE)
    eq = globalHistEq(img)
    cv2.imwrite('3-3/{}eq.jpg'.format(i), eq)

    median_filtered = median_filter(eq, 5)
    cv2.imwrite('3-3/{}median.jpg'.format(i), globalHistEq(median_filtered))

    edge_v = normalize(gradient(img, sobel_v))
    edge_h = normalize(gradient(img, sobel_h))
    edged = normalize(median_filtered + 0.8 * (edge_h + edge_v)).astype(np.uint8)
    cv2.imwrite('3-3/{}edge.jpg'.format(i), globalHistEq(edged))

    box_filtered = box_filter(edged, 3)
    cv2.imwrite('3-3/{}filtered.jpg'.format(i), globalHistEq(box_filtered))
