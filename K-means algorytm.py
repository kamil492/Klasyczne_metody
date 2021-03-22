# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:50:12 2021

@author: 01111585
"""
import progressbar
import numpy as np 
import cv2
from skimage import io
import os

img_path = "dane/images/"
img_list = os.listdir(img_path)
outPath = "wyniki_kmeans1/"
image_no = 1 

print("starting k-means algorithm")
for image in img_list:

    image_org = io.imread(os.path.join(img_path, image))
    img = cv2.imread(os.path.join(img_path, image))
    img2 = img.reshape((-1,3))
    #zmiana datatype
    
    img3 = np.float32(img2)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    #ilosc klas na obrazie
    k=2
    
    attempts = 12
    ret,label,center=cv2.kmeans(img3,k,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    name = 'wyniki_kmeans1/file_' + str(image_no) + '.jpg'
    cv2.imwrite(name, res2)
    image_no += 1

print("Done!")