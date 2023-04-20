import numpy as np


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    normalized = (img - min) / (max - min) * 255
    return normalized


def rect_to_polar(img_ft):
    amp = np.absolute(img_ft)
    phase = np.angle(img_ft)
    return amp, phase


def polar_to_rect(amp, phase):
    return amp * np.exp(1j * phase)


def log_ft(ft_img):
    res = np.log(ft_img + 1)
    res = normalize(res)
    return res
