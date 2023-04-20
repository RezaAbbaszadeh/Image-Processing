import numpy as np
import cv2
import statistics
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt

img_names = ['Original.bmp', '1.bmp', '2.bmp', '3.bmp', '4.bmp']
original = cv2.imread('Original.bmp', cv2.IMREAD_GRAYSCALE)
ref = cv2.imread('Reference.bmp', cv2.IMREAD_GRAYSCALE)
data = []
for img_name in img_names:
    img_attack1 = cv2.imread('Attack 1/{}'.format(img_name), cv2.IMREAD_GRAYSCALE)
    img_attack2 = cv2.imread('Attack 2/{}'.format(img_name), cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create(400)
    kp1, des1 = surf.detectAndCompute(ref, None)
    kp2, des2 = surf.detectAndCompute(img_attack1, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < 0.74 * m2.distance:
            good_matches.append([m1])
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)
    warped_image = cv2.warpPerspective(img_attack2, H, (ref.shape[1], ref.shape[0]))
    if img_name == '3.bmp':
        crop_start = 63
        crop_end = warped_image.shape[0] - crop_start

        cv2.imwrite('7-1/Attack-warped{}.jpg'.format(img_name),
                    warped_image[crop_start:crop_end, crop_start:crop_end])
        cv2.imwrite('7-1/Attack-diff{}.jpg'.format(img_name),
                    original[crop_start:crop_end, crop_start:crop_end] - warped_image[crop_start:crop_end, crop_start:crop_end])
        score, diff = compare_ssim(warped_image[crop_start:crop_end, crop_start:crop_end],
                                   original[crop_start: crop_end, crop_start: crop_end], full=True)
        mse = mean_squared_error(
            warped_image[crop_start:crop_end, crop_start:crop_end],
            original[crop_start:crop_end, crop_start:crop_end])
        mp = len(good_matches)
        data.append([img_name, score, mse, mp])
    else:
        cv2.imwrite('7-1/Attack-warped{}.jpg'.format(img_name), warped_image)
        cv2.imwrite('7-1/Attack-diff{}.jpg'.format(img_name), original - warped_image)
        score, diff = compare_ssim(warped_image, original, full=True)
        mse = mean_squared_error(warped_image, original)
        mp = len(good_matches)
        data.append([img_name, score, mse, mp])


npArray_data = (np.array(data))[:, 1:].astype(float)
mean = ["mean", statistics.mean(npArray_data[:, 0]),
        statistics.mean(npArray_data[:, 1]),
        statistics.mean(npArray_data[:, 2])]
std = ["std", statistics.stdev(npArray_data[:, 0]),
       statistics.stdev(npArray_data[:, 1]),
       statistics.stdev(npArray_data[:, 2])]
data.append(mean)
data.append(std)
types = [" ", "ssim", "mse", "mp"]
col_labels = tuple(types)
fig, ax = plt.subplots(dpi=300, figsize=(5, 5))
ax.axis('off')
ax.table(cellText=data, colLabels=col_labels, loc='center')
fig.savefig('7-1/res.png')
