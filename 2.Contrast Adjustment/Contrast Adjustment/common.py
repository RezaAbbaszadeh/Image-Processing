import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def add_padding(img, padSize, padValue=128):
    padded = np.ones((img.shape[0] + padSize * 2, img.shape[1] + padSize * 2), np.uint8) * padValue
    padded[padSize:img.shape[0] + padSize, padSize:img.shape[1] + padSize] = img
    return padded


def localHistEq(img, mainWindow=45, innerWindow=15):
    windowsDiff = mainWindow - innerWindow
    padding = int(windowsDiff / 2)
    img = add_padding(img, padding)
    res = np.zeros((img.shape[0] + windowsDiff, img.shape[1] + windowsDiff), np.uint8)
    for i in range(0, res.shape[0], innerWindow):
        if i + mainWindow > res.shape[0] - 1:
            i = res.shape[0] - mainWindow
        for j in range(0, res.shape[1], innerWindow):
            if j + mainWindow > res.shape[1] - 1:
                j = res.shape[1] - mainWindow
            pdf, cdf = compute_pdf_cdf(img[i:i + mainWindow, j:j + mainWindow])
            for i2 in range(i + padding, i + mainWindow - padding):
                for j2 in range(j + padding, j + mainWindow - padding):
                    res[i2][j2] = round(255 * cdf[img[i2 - padding][j2 - padding]])

    return res[windowsDiff:res.shape[0] - windowsDiff, windowsDiff:res.shape[1] - windowsDiff]


def normalize(img):
    return (((img - img.min()).astype(float) / (img.max() - img.min())) * 255).astype(np.uint8)
