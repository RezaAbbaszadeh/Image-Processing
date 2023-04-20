import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from common import compute_psnr


def cluster(img, levels):
    cluster_centers = []
    cluster_items = []
    for i in range(levels):
        cluster_items.append([])
    img_clusters = np.zeros((img.shape[0], img.shape[1]))
    for i in range(levels):
        #             blue                  green                   red
        t = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cluster_centers.append(t)

    any_cluster_updated = True
    counter = 0
    while any_cluster_updated:
        counter += 1
        print(str(counter))
        any_cluster_updated = False

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                min_distance = 999999
                min_distance_cluster = 0
                for cluster_index in range(len(cluster_centers)):
                    distance = np.sqrt(
                        ((img[i, j, 0] - cluster_centers[cluster_index][0]) ** 2) +
                        ((img[i, j, 1] - cluster_centers[cluster_index][1]) ** 2) +
                        ((img[i, j, 2] - cluster_centers[cluster_index][2]) ** 2))
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_cluster = cluster_index
                if img_clusters[i, j] != min_distance_cluster:
                    img_clusters[i, j] = min_distance_cluster
                    any_cluster_updated = True
                    cluster_items[min_distance_cluster].append(img[i, j, :])
        if not any_cluster_updated:
            break
        for cluster_index in range(len(cluster_items)):
            sum_rgb = [0, 0, 0]
            for item in cluster_items[cluster_index]:
                sum_rgb += item
            if len(cluster_items[cluster_index]) > 0:
                new_rgb = np.rint(sum_rgb / len(cluster_items[cluster_index]))
                cluster_centers[cluster_index] = (new_rgb[0], new_rgb[1], new_rgb[2])
    return cluster_centers, img_clusters

levels = [4, 8, 16, 32]
img = cv2.imread('Baboon.bmp')

PSNRs = ["psnr"]
MSEs = ["mmse"]
for level in levels:
    print("*********** {} ************".format(level))
    new_img = copy.deepcopy(img)
    kernel_mat, clusters = cluster(new_img, level)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j, :] = kernel_mat[int(clusters[i, j])]
    cv2.imwrite('5-2-3/res{}.png'.format(level), new_img)

    res = cv2.imread('5-2-3/res{}.png'.format(level))
    psnr, mmse = compute_psnr(img, res)
    MSEs.append(mmse)
    PSNRs.append(psnr)

data = [MSEs, PSNRs]
labels = (" ",) + tuple(levels)
fig, ax = plt.subplots(dpi=200, figsize=(7, 1))
ax.axis('off')
ax.table(colLabels=labels, cellText=data, loc='center')

fig.savefig('5-2-3/result.jpg')
