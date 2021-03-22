# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:17:43 2021

@author: Kamil Kowalczyk
"""
import os
import pickle
import numpy as np
from tqdm import tqdm 
from skimage.io import imread
from skimage.transform import resize

#ustalenie rozmiaru obrazow

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
IMG_CHANNELS_OTSU = 2
img_path = "wyniki_kmeans1/"
img_list = os.listdir(img_path)
imgOtsu_path = "wynikiOtsu/"
imgOtsu_list = os.listdir(imgOtsu_path)
mask_path = "dane/masks/"
mask_list = os.listdir(mask_path)
outPath = "wyniki"


Inputs_kmeans = np.zeros((len(img_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Inputs_otsu = np.zeros((len(img_list), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.uint8)
Y_train = np.zeros((len(mask_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

#zmiana rozmiaru i zaladowanie obrazow
print('Prepering images')
for n, id_ in tqdm(enumerate(img_list), total=len(img_list)):   
    path = img_path 
    # img = imread(path +  id_ )[:,:,:IMG_CHANNELS]
    # Inputs_kmeans[n] = img  
    img = imread(path +  id_ )[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    Inputs_kmeans[n] = img  

for n, id_ in tqdm(enumerate(imgOtsu_list), total=len(imgOtsu_list)):   
    path3 = imgOtsu_path
    Otsu = imread(path3 + id_ )
    img = np.expand_dims(resize(Otsu, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                       preserve_range=True), axis=-1)
    Inputs_otsu[n] = img  
    
    
for n, id_ in tqdm(enumerate(mask_list), total=len(mask_list)): 
    path2 = mask_path 
    mask = imread(path2 + id_ )
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                       preserve_range=True), axis=-1)
    Y_train[n] = mask   


#utworzenie zmiennych dla obrazow w formie pliku do odczytu
pickle_out = open("InputsKmeans256.pickle","wb")
pickle.dump(Inputs_kmeans, pickle_out)
pickle_out.close()

pickle_out = open("Ground_truth256.pickle","wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()

pickle_out = open("InputsOtsu256.pickle","wb")
pickle.dump(Inputs_otsu, pickle_out)
pickle_out.close()
