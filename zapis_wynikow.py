# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:55:38 2021

@author: Kamil Kowalczyk
"""

from scipy.spatial import distance
import numpy as np 
import cv2
from matplotlib import pyplot as plt
from skimage import data, io, img_as_ubyte
from skimage import color
import os

target = (color.rgb2gray(io.imread("images/testowyKmeans.jpg")))
target1 = (color.rgb2gray(io.imread("images/testowyOtsu.jpg")))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
ax[0].imshow(target1)
ax[0].set_title('Segmentacja Multi-Otsu')
ax[0].axis('off')

ax[1].imshow(target, cmap='gray')
ax[1].set_title('Segmentacja k-means')
ax[1].axis('off')

