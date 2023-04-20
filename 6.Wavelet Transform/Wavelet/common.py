import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    normalized = (img - min) / (max - min) * 255
    return normalized


def psnr(im1, im2):
    mse = np.mean((im1 - im2) ** 2)
    if mse == 0:
        return np.inf, 0
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr, mse


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), .1)


def average_blur(img):
    return cv2.blur(img, (2, 2))


def approximation_pyramid(img, level, mode='gaussian'):
    pyramid_list = [img]
    gauss = img
    for i in range(level):
        gauss = average_blur(gauss) if (mode == 'average') else gaussian_blur(gauss)
        gauss = cv2.resize(gauss, (int(gauss.shape[1] / 2), int(gauss.shape[0] / 2)))
        pyramid_list.append(gauss)
    return pyramid_list


def up_sample(img, target_shape):
    result = np.ones(target_shape)
    is_height_odd = 0 if (target_shape[0] % 2 == 0) else 1
    is_width_odd = 0 if (target_shape[1] % 2 == 0) else 1
    for i in range(0, target_shape[0] - is_height_odd):
        for j in range(0, target_shape[1] - is_width_odd):
            result[i, j] = img[int(i / 2), int(j / 2)]
    return result


def laplacian_pyramid(imgs):
    pyramid_list = []
    for i in range(1, len(imgs)):
        upsample = up_sample(imgs[i], imgs[i - 1].shape)
        laplace = imgs[i - 1] - upsample
        pyramid_list.append(laplace)
    pyramid_list.append(imgs[-1])
    return pyramid_list


def reconstruct_from_laplacian(pyramid, img):
    upsample = copy.deepcopy(img)
    for i in range(len(pyramid) - 2, -1, -1):
        upsample = up_sample(upsample, pyramid[i].shape)
        upsample = upsample + pyramid[i]
    return upsample


def show_wt(cA, cH, cV, cD, file_name):
    plt.figure(figsize=(30, 30))

    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap=plt.cm.gray)
    plt.title('cA: ', fontsize=50)

    plt.subplot(2, 2, 2)
    plt.imshow(cH, cmap=plt.cm.gray)
    plt.title('cH: ', fontsize=50)

    plt.subplot(2, 2, 3)
    plt.imshow(cV, cmap=plt.cm.gray)
    plt.title('cV: ', fontsize=50)

    plt.subplot(2, 2, 4)
    plt.imshow(cD, cmap=plt.cm.gray)
    plt.title('cD: ', fontsize=50)
    plt.savefig(file_name)
