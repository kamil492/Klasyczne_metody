# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:08:49 2021

@author: 01111585
"""

import progressbar
import numpy as np 
import cv2
from skimage import io
from skimage import color
import os
from skimage.filters import threshold_multiotsu



img_path = "dane/images/"
img_list = os.listdir(img_path)
outPath = "wyniki"
image_no = 1 
print("starting multi-Otsu algorithm")
for image in (img_list):
    # wczytanie obrazu
    image = (color.rgb2gray(io.imread(os.path.join(img_path, image))))
    # wywolanie funkcji multiotsu z biblioteki scikit image 
    thresholds = threshold_multiotsu(image, classes=2)
    
    #zamiana danych z binarnych na postac cyfrowa
    regions = np.digitize(image, bins=thresholds)
    
    # konwersja 64bit na 8 bti z normalizacja etc
    img_n = cv2.normalize(src=regions, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    output = img_n.astype(np.uint8)
    name = 'wynikiOtsu/file_' + str(image_no) + '.jpg'
    cv2.imwrite(name, output)
    image_no += 1

print("Done!")
