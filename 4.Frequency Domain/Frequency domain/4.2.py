import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import rect_to_polar, log_ft, normalize, polar_to_rect

def generate_filter(f_name, T, N):
    filter = np.ones((N, N))
    if f_name == 'a':
        filter[int(T * N):int((1 - T) * N), int(T * N):int((1 - T) * N)] = 0
    elif f_name == 'b':
        filter[0:int(T * N), 0:int(T * N) + 1] = 0
        filter[0:int(T * N), int((1 - T) * N):N] = 0
        filter[int((1 - T) * N):N, 0:int(T * N) + 1] = 0
        filter[int((1 - T) * N):N, int((1 - T) * N):N] = 0
    return filter


img = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)
N = img.shape[0]

img_ft = np.fft.fft2(img)
img_amp, img_phase = rect_to_polar(img_ft)
cv2.imwrite('Lena-Ft.jpg', log_ft(img_amp))
Ts = [1 / 4, 1 / 8]
filter_names = ['a', 'b']

for f_name in filter_names:
    for t in Ts:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        H = generate_filter(f_name, t, N)

        axs[0].imshow(normalize(H), cmap='gray', aspect='auto')
        axs[0].set_title('filter')

        RES = img_amp * H

        RES = polar_to_rect(RES, img_phase)
        res = np.fft.ifft2(RES)
        res = res.real

        axs[1].imshow(normalize(res), cmap='gray', aspect='auto')
        axs[1].set_title('result')

        fig.savefig('4.2.1/H.{}-T.{}.jpg'.format(f_name, t), bbox_inches='tight')
