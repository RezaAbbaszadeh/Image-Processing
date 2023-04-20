import numpy as np


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = (img - min) / (max - min) * 255
    return img


def add_padding(img, padSize, padValue=0):
    padded = np.ones((img.shape[0] + padSize * 2, img.shape[1] + padSize * 2), np.uint8) * padValue
    padded[padSize:img.shape[0] + padSize, padSize:img.shape[1] + padSize] = img
    return padded


def cut_padding(img, padSize):
    return img[padSize: img.shape[0] - padSize,
           padSize: img.shape[1] - padSize]


def convolve_filter(img, window):
    padding_size = int(window.shape[0] / 2)
    padded_img = add_padding(img, padding_size)
    filtered = np.zeros(padded_img.shape, np.uint8)
    sum_of_filter = sum(window.flatten())

    for i in range(padding_size, img.shape[0] + padding_size):
        for j in range(padding_size, img.shape[1] + padding_size):
            filtered[i, j] = (1 / sum_of_filter) * sum(
                np.multiply(
                    np.asarray(padded_img[i - padding_size:i + padding_size + 1,
                               j - padding_size:j + padding_size + 1]
                               ),
                    window
                )
                    .flatten()
            )

    return cut_padding(filtered, padding_size)


def box_filter(img, window_size):
    return convolve_filter(img, np.ones((window_size, window_size)))


def laplacian_filter(img, window):
    return convolve_filter(img, window)


def median_filter(img, windowSize):
    padding_size = int(windowSize / 2)
    padded_img = add_padding(img, padding_size)
    filtered = np.zeros(padded_img.shape, np.uint8)

    for i in range(padding_size, img.shape[0] + padding_size):
        for j in range(padding_size, img.shape[1] + padding_size):
            filtered[i, j] = np.median(np.asarray(padded_img[i - padding_size:i + padding_size + 1,
                                                  j - padding_size:j + padding_size + 1]
                                                  ))
    return cut_padding(filtered, padding_size)


def compute_histogram(img):
    histogram = np.zeros(256)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            histogram[img[i, j]] += 1
    return histogram


def compute_pdf_cdf(img):
    histogram = compute_histogram(img)

    pdf = histogram / (img.shape[0] * img.shape[1])
    cdf = pdf
    for i in range(1, 256):
        cdf[i] = cdf[i] + cdf[i - 1]

    return pdf, cdf


def globalHistEq(img):
    pdf, cdf = compute_pdf_cdf(img)

    res = np.zeros(img.shape, np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            res[i][j] = round(255 * cdf[img[i][j]])

    return res


def unsharp_masking(img, k, window):
    smoothed = convolve_filter(img, window)
    mask = img - smoothed
    res = img + k * mask
    return normalize(res)

