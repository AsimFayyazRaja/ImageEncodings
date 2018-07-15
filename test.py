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

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import load_model


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
            '''
            if flag and img!="resized/si.jpg":
                print("passed")
                continue
            '''
            if flag:
                im=img[19:-4]
            else:
                im=img[15:-4]
            names.append(im)
            n= cv2.imread(img)
            bag.append(n)
            labels.append([1,1,1,1,1,1,1,1,1,1])
            pbar.update(1)
            r+=1
    return bag,labels,names

X,y,names=generate_training_data("ms-coco_resized128",True)

X_test,y1,test_names=generate_training_data("test_resized64",False)

X_test=np.array(X_test)
#print(X_test.shape)
X=np.array(X)
y=np.array(y)
#print(X.shape)
#print(y.shape)
#y=np.reshape(y,[-1,3])

model = load_model('keras_model/model.h5')

temp=[]
arr=[]
test_encodings=[]
i=0
k=0
with tqdm(total=len(X_test)) as pbar:
    for x in X_test:
        k=0
        for im in X:
            arr=[]
            im=np.array(im)
            M = im.shape[0]//2
            N = im.shape[1]//2
            tiles = [im[x:x+M,y:y+N] for x in range(0,im.shape[0],M) for y in range(0,im.shape[1],N)]
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'img1.jpg',tiles[0])
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'test_img.jpg',x)
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'img2.jpg',tiles[1])
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'img3.jpg',tiles[2])
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'img4.jpg',tiles[3])
            
            res=np.vstack((x,tiles[2]))
            res1=np.vstack((tiles[1],tiles[3]))
            res3=np.hstack((res,res1))
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'test1.jpg',res3)
            res3=np.expand_dims(res3,axis=0)
            model_out=model.predict([res3])[0]
            arr.append([model_out[0],model_out[1],model_out[2],model_out[3]
            ,model_out[4],model_out[5],model_out[6],model_out[7],model_out[8]
            ,model_out[9]])

            res=np.vstack((tiles[0],x))
            res1=np.vstack((tiles[1],tiles[3]))
            res3=np.hstack((res,res1))
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'test2.jpg',res3)
            res3=np.expand_dims(res3,axis=0)
            model_out=model.predict([res3])[0]
            arr.append([model_out[0],model_out[1],model_out[2],model_out[3]
            ,model_out[4],model_out[5],model_out[6],model_out[7],model_out[8]
            ,model_out[9]])

            res=np.vstack((tiles[0],tiles[2]))
            res1=np.vstack((x,tiles[3]))
            res3=np.hstack((res,res1))
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'test3.jpg',res3)
            res3=np.expand_dims(res3,axis=0)
            model_out=model.predict([res3])[0]
            arr.append([model_out[0],model_out[1],model_out[2],model_out[3]
            ,model_out[4],model_out[5],model_out[6],model_out[7],model_out[8]
            ,model_out[9]])


            res=np.vstack((tiles[0],tiles[2]))
            res1=np.vstack((tiles[1],x))
            res3=np.hstack((res,res1))
            #cv2.imwrite('temp/'+test_names[i]+names[k]+'test4.jpg',res3)
            res3=np.expand_dims(res3,axis=0)
            model_out=model.predict([res3])[0]
            arr.append([model_out[0],model_out[1],model_out[2],model_out[3]
            ,model_out[4],model_out[5],model_out[6],model_out[7],model_out[8]
            ,model_out[9]])
            #nm=str(test_names[i]+names[i])
            j=0
            for j in range(4):
                test_encodings.append([test_names[i],names[k],j+1,arr[j][0],arr[j][1]
                ,arr[j][2],arr[j][3],arr[j][4],arr[j][5],arr[j][6],arr[j][7]
                ,arr[j][8],arr[j][9]])
            k+=1
            pbar.update(1)
        i+=1
        
csvfile = "test_encodings.csv"
print("Writing test encodings to csv file")
i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["TestImage","TrainImage","Chunk","value1","value2","value3",
        "value4","value5","value6","value7","value8","value9","value10"])
    i+=1
    writer.writerows(test_encodings)

print("data dumped to file")















