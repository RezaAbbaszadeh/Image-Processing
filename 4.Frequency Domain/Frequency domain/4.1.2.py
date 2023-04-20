import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import rect_to_polar, normalize

images = ['Lena', 'Barbara', 'F16', 'Baboon']

for img_name in images:
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    img = cv2.imread(img_name + ".bmp", cv2.IMREAD_GRAYSCALE)
    axs[0][0].imshow(img, cmap='gray', aspect='auto')
    axs[0][0].set_title('Original image')
    axs[0][0].axis('off')

    img_ft = np.fft.fft2(img)
    img_amp, img_phase = rect_to_polar(img_ft)
    axs[0][1].imshow(img_amp, cmap='gray', aspect='auto')
    axs[0][1].set_title('DFT')
    axs[0][1].axis('off')

    logged = normalize(np.log(img_amp + 1))
    logged = normalize(logged)
    axs[1][0].imshow(logged, cmap='gray', aspect='auto')
    axs[1][0].set_title('logged DFT')
    axs[1][0].axis('off')

    shifted = np.fft.fftshift(logged)
    axs[1][1].imshow(shifted, cmap='gray', aspect='auto')
    axs[1][1].set_title('shifted logged DFT')
    axs[1][1].axis('off')
    fig.savefig('4.1.2/{}-FT.png'.format(img_name), bbox_inches='tight')
