# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:33:02 2020

@author: Kamil Kowalczyk
"""

from scipy.spatial import distance
import numpy as np 
import cv2
from matplotlib import pyplot as plt
from skimage import data, io, img_as_ubyte
from skimage import color
import os

## metryki - policzenie poprawnosci dla pojedynczego obrazu
def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union
def iou_score(inputs,target):
    intersection = np.logical_and(inputs, target)
    union = np.logical_or(inputs, target)
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return np.sum(intersection) / np.sum(union)

img_path = "data\images"
img_list = os.listdir(img_path)
mask_path = "data\masks"
mask_list = os.listdir(mask_path)
outPath = "wyniki"



image_org = io.imread("images/testorg.jpg")
img = cv2.imread("images/testorg.jpg") 
img_m = cv2.imread("images/mask_test.jpg") 
plt.imshow(img)
target = (color.rgb2gray(io.imread("images/mask_test.jpg")))
img2 = img.reshape((-1,3))
img3 = img_m.reshape((-1,3))

#zmiana datatype

img3 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

#ilosc klas 
k=2

attempts = 12
ret,label,center=cv2.kmeans(img3,k,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
plt.imsave("images/testowyKmeans.jpg", res2, cmap='gray')
# cv2.imwrite('images/treningowyKmeans.jpg', res2)
plt.imshow(res2)


#miary poprawnosci segmentacji
iouscore = iou_score(res2,img_m)
score = dice_metric(res2,img_m)

##wyswietlenie wynikow
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax[0].imshow(image_org)
ax[0].set_title('Obraz oryginalny')
ax[0].axis('off')

ax[1].imshow(target, cmap='gray')
ax[1].set_title('Maska obrazu')
ax[1].axis('off')

ax[2].imshow(res2)
ax[2].set_title('Segmentacja k-means')
ax[2].axis('off')

