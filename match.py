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
import os
import time
import imageio
from skimage.io import imsave
from skimage import img_as_ubyte
from skimage import img_as_float
import cv2
import ast

class DictList(dict):
    def __setitem__(self, key, value):
        try:
            # Assumes there is a list on the key
            self[key].append(value) 
        except KeyError: # if fails because there is no key
            super(DictList, self).__setitem__(key, value)
        except AttributeError: # if fails because it is not a list
            super(DictList, self).__setitem__(key, [self[key], value])


def get_dataset_encodings():
    df=pd.read_csv('dataset_encodings.csv')       #getting file
    print("Loading dataset encodings..")
    with tqdm(total=len(list(df.iterrows()))) as prbar:
        test_encodings=DictList()
        for index, row in df.iterrows():
            img=df.loc[index,'Img_Name']
            value1=float(df.loc[index,'value1'])
            value2=float(df.loc[index,'value2'])
            value3=float(df.loc[index,'value3'])
            value4=float(df.loc[index,'value4'])
            value5=float(df.loc[index,'value5'])
            value6=float(df.loc[index,'value6'])
            value7=float(df.loc[index,'value7'])
            value8=float(df.loc[index,'value8'])
            value9=float(df.loc[index,'value9'])
            value10=float(df.loc[index,'value10'])
            
            encodings=[value1,value2,value3,value4,value5,value6,value7
            ,value8,value9,value10]
            test_encodings[img]=encodings
            prbar.update(1)
    return test_encodings


def get_test_encodings():
    df=pd.read_csv('test_encodings.csv')       #getting file
    print("Loading test encodings..")
    with tqdm(total=len(list(df.iterrows()))) as prbar:
        test_encodings=[]
        for index, row in df.iterrows():
            trainimg=df.loc[index,'TrainImage']
            testimg=df.loc[index,'TestImage']
            value1=float(df.loc[index,'value1'])
            value2=float(df.loc[index,'value2'])
            value3=float(df.loc[index,'value3'])
            value4=float(df.loc[index,'value4'])
            value5=float(df.loc[index,'value5'])
            value6=float(df.loc[index,'value6'])
            value7=float(df.loc[index,'value7'])
            value8=float(df.loc[index,'value8'])
            value9=float(df.loc[index,'value9'])
            value10=float(df.loc[index,'value10'])
            
            encodings=[value1,value2,value3,value4,value5,value6,value7
            ,value8,value9,value10]
            test_encodings.append([testimg,trainimg,encodings])
            prbar.update(1)
    return test_encodings


dataset_encodings=get_dataset_encodings()   #dataset encodings here as dict
#print(dataset_encodings)

test_encodings=get_test_encodings()     #test encodings here as dict
#print(test_encodings)
arr=DictList()
norm1=DictList()
norm2=DictList()
norm3=DictList()
norm4=DictList()
arrn1=[]
arrn2=[]
arrn3=[]
arrn4=[]
print("Calculating norms..")
with tqdm(total=len(test_encodings)) as pbar:
    for enc in test_encodings:
        test1=np.array(enc[2])
        train=np.array(dataset_encodings[enc[1]])
        norm=np.linalg.norm(train-test1)
        arr[enc[0]+"-"+enc[1]]=norm
        pbar.update(1)

towrite=[]
print("Getting shortest norms..")
with tqdm(total=len(arr)) as pbar:
    for k,v in arr.items():
        k=str(k)
        t=k.split('-')
        testimg=t[0]
        trainimg=t[1]
        temp=[testimg,trainimg,v[0],v[1],v[2],v[3]]
        towrite.append(temp)
        arrn1.append(v[0])
        arrn2.append(v[1])
        arrn3.append(v[2])
        arrn4.append(v[3])
        norm1[v[0]]=[testimg,trainimg]
        norm2[v[1]]=[testimg,trainimg]
        norm3[v[2]]=[testimg,trainimg]
        norm4[v[3]]=[testimg,trainimg]
        pbar.update(1)

csvfile = "norms.csv"
print("Writing norms to csv file")
i=0
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    if(i==0):
        writer.writerow(["TestImage","TrainImage","Norm1","Norm2","Norm3","Norm4"])
    i+=1
    writer.writerows(towrite)

print("data dumped to file")

m1=min(arrn1)
m2=min(arrn2)
m3=min(arrn3)
m4=min(arrn4)
print(m1,norm1[m1])
print(m2,norm2[m2])
print(m3,norm3[m3])
print(m4,norm4[m4])
