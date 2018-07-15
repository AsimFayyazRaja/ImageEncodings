import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
import random
from random import randint
from sklearn.metrics import average_precision_score
import pandas as pd
from scipy import misc as cv
import glob
import tensorflow as tf
from PIL import Image
from skimage import transform
import copy
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import os
import time
import imageio
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage import img_as_float
import cv2
#import image_slicer

from sklearn.neighbors import NearestNeighbors

def generate_training_data(folder,flag):
    r=0
    "Gets images for training and returns training data"
    print("Resizing images..")
    training_data = []
    bag=[]
    labels=[]
    names=[]
    with tqdm(total=len(glob.glob(folder+"/*.*"))) as pbar:
        for img in glob.glob(folder+"/*.*"):
            temp=[]
            if r>=300:
                break
            print(img)
            #if flag and img!="resized/si.jpg":
            #    print("passed")
            #    continue
            if flag:
                im=img[8:-4]
                print(im)
            else:
                im=img[13:-4]
                print(im)
            names.append(im)
            n= cv2.imread(img)
            bag.append(n)
            labels.append([1,1,1,1,1,1,1,1,1,1])
            pbar.update(1)
            r+=1
    return bag,labels,names

X,y,names=generate_training_data("resized",True)

X_test,y1,test_names=generate_training_data("test_resized128",False)
X+=X_test
names+=test_names
X=np.array(X)
print(X.shape)
X=np.reshape(X,[-1,128*128*3])
print(X.shape)
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)
print(names)
'''
csvfile = "siamese_test_encodings.csv"
print("Writing test encodings to csv file")
i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["TestImage","TrainImage","Chunk","Encodings"])
    i+=1
    writer.writerows(test_encodings)

print("data dumped to file")
'''













