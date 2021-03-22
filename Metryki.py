# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:58:06 2021

@author: 01111585
"""
import pickle
import numpy as np
import tensorflow as tf


#zdefiniowanie uzywanych metryk 

def jaccard_index_numpy(y_true, y_pred):
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)
    if (TP + FP + FN) == 0:
        jac = 0
    else:
        jac = TP / (TP + FP + FN)

    return jac
     

def dice_score(y_true, y_pred):
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)
    if (TP + FP + FN) == 0:
        dice = 0
    else:
        dice = 2*TP / (2*TP + FP + FN)
    return dice

def accuracy(y_true, y_pred):
    TP = np.count_nonzero(y_pred * y_true)
    FP = np.count_nonzero(y_pred * (y_true - 1))
    FN = np.count_nonzero((y_pred - 1) * y_true)
    TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
    if (TP + FP + FN) == 0:
        accuracy = 0
    else:
        accuracy = (TP+TN) / (TP+TN+FP+FN)
    return accuracy



#otwarcie zmiennych z obrazami
pickle_in = open("dane/InputsKmeans256.pickle","rb")
InputsKmeans = pickle.load(pickle_in)

pickle_in = open("dane/Ground_truth256.pickle","rb")
ground_truth = pickle.load(pickle_in)

pickle_in = open("dane/InputsOtsu256.pickle","rb")
InputsOtsu = pickle.load(pickle_in)


#obliczenie metryk dla poszczegolnych algorytmow
print('Calculating scores for k-means')
score = jaccard_index_numpy(ground_truth, InputsKmeans)
score_dice = dice_score(ground_truth, InputsKmeans)
accuracy = accuracy(ground_truth, InputsKmeans)
print('Done!')

print('Calculating scores for Multi-Otsu')
scoreOtsu = jaccard_index_numpy(ground_truth, InputsOtsu)
scoreOtsu_dice = dice_score(ground_truth, InputsOtsu)
accuracy1 = accuracy(ground_truth, InputsOtsu)
print('Done!')


