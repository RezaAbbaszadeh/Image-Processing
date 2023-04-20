import cv2
import matplotlib.pyplot as plt
import numpy as np


def quantize(img, level):
    return np.floor(img / (256 / level)).astype(np.uint8)


img = cv2.imread('Barbara.bmp', cv2.IMREAD_GRAYSCALE)

table_rows = [
    ["quantized"],
    ["equalized + quantized"]
]
table_cols = ("level", 4, 8, 16, 32, 64, 128)

for level in (4, 8, 12, 32, 64, 128):
    quantized = quantize(img, level)
    equalized_quantized = cv2.equalizeHist(quantized)

    mse_quantized = np.square(np.subtract(img, quantized)).mean()
    mse_equalized_quantized = np.square(np.subtract(img, equalized_quantized)).mean()

    table_rows[0].append(mse_quantized)
    table_rows[1].append(mse_equalized_quantized)

    fig1, axs = plt.subplots(2, 2)
    axs[0][0].imshow(quantized, cmap='gray')
    axs[0][0].set_title("without equalization")
    axs[1][0].hist(quantized.flatten(), 255)

    axs[0][1].imshow(equalized_quantized, cmap='gray')
    axs[0][1].set_title("with equalization")
    axs[1][1].hist(equalized_quantized.flatten(), 255)

    fig1.savefig('images/res' + str(level) + '.jpg')

fig, ax = plt.subplots(dpi=300, figsize=(7, 1))
ax.axis('off')
ax.table(cellText=table_rows, colLabels=table_cols, loc='center')
fig.savefig('images/table.jpg')
