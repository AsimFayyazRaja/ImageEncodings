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


def generate_training_data(folder,path):
    r=0
    "Gets images for training and returns training data"
    print("Resizing images..")
    training_data = []
    with tqdm(total=len(glob.glob(folder+"/*.*"))) as pbar:
        for img in glob.glob(folder+"/*.*"):
            temp=[]
            n= cv2.imread(img)
            n = cv2.resize(n, dsize=(64,64))
            path1=path+str(r)+".jpg"
            cv2.imwrite(path1,n)
            pbar.update(1)
            n=np.array(n)
            print(n)
            r+=1

generate_training_data("test","test_resized64/image")



