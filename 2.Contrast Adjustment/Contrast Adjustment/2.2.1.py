import cv2
from common import localHistEq

# [mainWindow, innerWindow]
window_sizes = [
    [200, 20],
    [100, 100],
    [100, 10],
    [50, 8],
    [30, 6],
]

for i in range(1, 5):
    img = cv2.imread('HE%i.jpg' % i, cv2.IMREAD_GRAYSCALE)
    for window in window_sizes:
        local_eq = localHistEq(img, window[0], window[1])
        cv2.imwrite("2.2.1-HE%i-%i-%i.jpg" % (i, window[0], window[1]), local_eq)
        print("2.2.1-HE%i-%i-%i.jpg saved" % (i, window[0], window[1]))
