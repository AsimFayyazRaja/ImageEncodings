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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import load_model


def generate_training_data(folder):
    r=0
    "Gets images for training and returns training data"
    print("Getting images..")
    training_data = []
    bag=[]
    labels=[]
    names=[]
    with tqdm(total=len(glob.glob(folder+"/*.*"))) as pbar:
        for img in glob.glob(folder+"/*.*"):
            temp=[]
            im=img[19:-4]
            names.append(im)
            n= np.array(cv2.imread(img))
            #n=n/127.5-1   #rescaling to -1,1
            bag.append(n)
            labels.append([1,1,1,1,1,1,1,1,1,1])
            pbar.update(1)
            r+=1
    return bag,labels,names

X,y,names=generate_training_data("ms-coco_resized128")
X=np.array(X)
y=np.array(y)
#y=np.reshape(y,[-1,3])

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(10, activation='relu'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X,y,
          batch_size=64,
          epochs=5,
          verbose=1)
'''
#model.save('keras_model/model.h5')

model = load_model('keras_model/model.h5')

arr=[]
sub=[]
i=0
print("Generating encodings..")
with tqdm(total=len(X)) as pbar:
    for x in X:
        #add expand dims to make it 4d
        x=np.expand_dims(x,axis=0)
        #print(x.shape)
        model_out=model.predict([x])[0]
        arr.append([names[i],model_out[0],model_out[1],model_out[2],model_out[3]
        ,model_out[4],model_out[5],model_out[6],model_out[7],model_out[8]
        ,model_out[9]])
        i+=1
        pbar.update(1)
csvfile = "dataset_encodings.csv"
print("Writing encodings to csv file")
i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["Img_Name","value1","value2","value3","value4","value5"
        ,"value6","value7","value8","value9","value10"])
    i+=1
    writer.writerows(arr)

print("data dumped to file")






















