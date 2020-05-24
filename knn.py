# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:38:50 2020

@author: fericjf
"""

import numpy as np
from scipy.stats import mode
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *


#load the facial recognition data
xTr,yTr,xTe,yTe=loaddata("faces.mat")


#plot the facial recognition data
plt.figure(figsize=(11,8))
plotfaces(xTr[:9, :])

#define knn function
def findknn(xTr,xTe,k):
    euclid_dist = l2distance(xTr, xTe)
    indices = np.argsort(euclid_dist, axis=0)
    indices = indices[:k]
    dists = np.sort(euclid_dist, axis=0)
    dists = dists[:k]
    
    return indices, dists

#show knn 2-d plots
visualize_knn_2D(findknn)

#show knn output as facial recognition format
visualize_knn_images(findknn, imageType='faces')



#define accuracy function
def accuracy(truth,preds):
    truth = truth.flatten()
    preds = preds.flatten()
    compareVectors = truth == preds
    compareVectors = compareVectors.astype(int)
    accuracy = np.sum(compareVectors)/len(truth)
    return accuracy

#define knn classifier outputting label
def knnclassifier(xTr,yTr,xTe,k):
    yTr = yTr.flatten()
    #import the indices value from findknn I wrote
    indices = findknn(xTr,xTe,k)
    indices = indices[0]
    indices = np.transpose(indices)

    #define preds before loop
    preds = []

    for i in range(len(xTe)):
        #look at ith value of indices to use for ytr
        ind_indices = indices[i]
        #using the index for yTr
        ind_y = np.ndarray.flatten(yTr[ind_indices])
        #use the mode function from scipy
        mode_y = mode(ind_y)[0]
        preds.append(mode_y)
        
    preds = np.ndarray.flatten(np.array(preds))
    return preds


print("Face Recognition: (1-nn)")
xTr,yTr,xTe,yTe=loaddata("faces.mat") # load the data
t0 = time.time()
preds = knnclassifier(xTr,yTr,xTe,1)
result=accuracy(yTe,preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result*100.0,t1-t0))


visualize_knn_boundary(knnclassifier)