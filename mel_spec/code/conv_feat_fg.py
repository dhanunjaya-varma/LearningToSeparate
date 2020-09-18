#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:20:48 2019

@author: dhanunjaya
"""

import os
import numpy as np
from keras.applications import VGG16
import cv2

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

input_path = '../feat/cross_val/foreground/fold1/train/' 
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold1_train = []
y_fold1_train = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold1_train.append(np.squeeze(conv_base.predict(img)))
        y_fold1_train.append(classes.index(clas))


x_fold1_train = np.asarray(x_fold1_train)
y_fold1_train = np.asarray(y_fold1_train)
x_fold1_train = np.reshape(x_fold1_train, (x_fold1_train.shape[0], x_fold1_train.shape[1]*x_fold1_train.shape[2]*x_fold1_train.shape[3]))
np.save("../data/x_foreground_fold1_train.npy", x_fold1_train)
np.save("../data/y_foreground_fold1_train.npy", y_fold1_train)  

input_path = '../feat/cross_val/foreground/fold1/test/' 
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold1_test = []
y_fold1_test = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold1_test.append(np.squeeze(conv_base.predict(img)))
        y_fold1_test.append(classes.index(clas))


x_fold1_test = np.asarray(x_fold1_test)
y_fold1_test = np.asarray(y_fold1_test)
x_fold1_test = np.reshape(x_fold1_test, (x_fold1_test.shape[0], x_fold1_test.shape[1]*x_fold1_test.shape[2]*x_fold1_test.shape[3]))
np.save("../data/x_foreground_fold1_test.npy", x_fold1_test)
np.save("../data/y_foreground_fold1_test.npy", y_fold1_test)  

########fold2

input_path = '../feat/cross_val/foreground/fold2/train/' 
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold2_train = []
y_fold2_train = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold2_train.append(np.squeeze(conv_base.predict(img)))
        y_fold2_train.append(classes.index(clas))


x_fold2_train = np.asarray(x_fold2_train)
y_fold2_train = np.asarray(y_fold2_train)
x_fold2_train = np.reshape(x_fold2_train, (x_fold2_train.shape[0], x_fold2_train.shape[1]*x_fold2_train.shape[2]*x_fold2_train.shape[3]))
np.save("../data/x_foreground_fold2_train.npy", x_fold2_train)
np.save("../data/y_foreground_fold2_train.npy", y_fold2_train)  

input_path = '../feat/cross_val/foreground/fold2/test/' 
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold2_test = []
y_fold2_test = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold2_test.append(np.squeeze(conv_base.predict(img)))
        y_fold2_test.append(classes.index(clas))


x_fold2_test = np.asarray(x_fold2_test)
y_fold2_test = np.asarray(y_fold2_test)
x_fold2_test = np.reshape(x_fold2_test, (x_fold2_test.shape[0], x_fold2_test.shape[1]*x_fold2_test.shape[2]*x_fold2_test.shape[3]))
np.save("../data/x_foreground_fold2_test.npy", x_fold2_test)
np.save("../data/y_foreground_fold2_test.npy", y_fold2_test) 

####fold3

input_path = '../feat/cross_val/foreground/fold3/train/' 
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold3_train = []
y_fold3_train = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold3_train.append(np.squeeze(conv_base.predict(img)))
        y_fold3_train.append(classes.index(clas))


x_fold3_train = np.asarray(x_fold3_train)
y_fold3_train = np.asarray(y_fold3_train)
x_fold3_train = np.reshape(x_fold3_train, (x_fold3_train.shape[0], x_fold3_train.shape[1]*x_fold3_train.shape[2]*x_fold3_train.shape[3]))
np.save("../data/x_foreground_fold3_train.npy", x_fold3_train)
np.save("../data/y_foreground_fold3_train.npy", y_fold3_train)  

input_path = '../feat/cross_val/foreground/fold3/test/' 
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold3_test = []
y_fold3_test = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold3_test.append(np.squeeze(conv_base.predict(img)))
        y_fold3_test.append(classes.index(clas))


x_fold3_test = np.asarray(x_fold3_test)
y_fold3_test = np.asarray(y_fold3_test)
x_fold3_test = np.reshape(x_fold3_test, (x_fold3_test.shape[0], x_fold3_test.shape[1]*x_fold3_test.shape[2]*x_fold3_test.shape[3]))
np.save("../data/x_foreground_fold3_test.npy", x_fold3_test)
np.save("../data/y_foreground_fold3_test.npy", y_fold3_test) 

###fold4
input_path = '../feat/cross_val/foreground/fold4/train/'
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold4_train = []
y_fold4_train = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold4_train.append(np.squeeze(conv_base.predict(img)))
        y_fold4_train.append(classes.index(clas))


x_fold4_train = np.asarray(x_fold4_train)
y_fold4_train = np.asarray(y_fold4_train)
x_fold4_train = np.reshape(x_fold4_train, (x_fold4_train.shape[0], x_fold4_train.shape[1]*x_fold4_train.shape[2]*x_fold4_train.shape[3]))
np.save("../data/x_foreground_fold4_train.npy", x_fold4_train)
np.save("../data/y_foreground_fold4_train.npy", y_fold4_train)  

input_path = '../feat/cross_val/foreground/fold4/test/'
#output='/home/dhanunjaya/Documents/DCASE17/mel_spec/audio/'

x_fold4_test = []
y_fold4_test = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        img = cv2.imread(filePath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        x_fold4_test.append(np.squeeze(conv_base.predict(img)))
        y_fold4_test.append(classes.index(clas))


x_fold4_test = np.asarray(x_fold4_test)
y_fold4_test = np.asarray(y_fold4_test)
x_fold4_test = np.reshape(x_fold4_test, (x_fold4_test.shape[0], x_fold4_test.shape[1]*x_fold4_test.shape[2]*x_fold4_test.shape[3]))
np.save("../data/x_foreground_fold4_test.npy", x_fold4_test)
np.save("../data/y_foreground_fold4_test.npy", y_fold4_test)