# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 09:52:14 2017

@author: PN JUNCTION LAB
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
#import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 32, 32

# number of channels
img_channels = 3

#%%

PATH=os.getcwd()
dirname = 'data'
dim=(img_rows, img_cols)    #path of folder of images    
#path2 = 'images_resized//'  #path of folder to save images  


immatrix=[]
length_dir=[]
for class_directory in os.listdir(dirname):
    length_dir.append([])
    if os.path.isdir(os.path.join(dirname, class_directory)): 
        for filename in os.listdir(os.path.join(dirname,class_directory)): 
            
            img_path = os.path.join(dirname, class_directory, filename)

            img = Image.open(img_path).resize((dim))
            
            img = np.array(img).flatten()
            #img.reshape(32,32,3)
            immatrix.append(img.astype('float32'))
            

immatrix = np.asarray(immatrix)


length_dir=[]
for class_directory in os.listdir(dirname):
    length_dir.append(len(os.listdir(os.path.join(dirname,class_directory))))
#%%
           
# 
num_samples=0
for i in range(len(length_dir)):
    num_samples+=length_dir[i] 

array=[]
array.append(0)
k=0
for i in range(len(length_dir)):
    k+=length_dir[i]
    array.append(k)
           
label=np.ones((num_samples,),dtype = int)

for i in range(len(array)-1):
    label[array[i]:array[i+1]]=i
    
k= len(array)-1
label[array[k]:]=k   

#%%
data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]


#batch_size = 32
# number of output classes
#nb_classes = 4
# number of epochs to train
nb_epoch = 100


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X, y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols,3)
X_test = X_test.reshape(X_test.shape[0],  img_rows, img_cols,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, k)
Y_test = np_utils.to_categorical(y_test, k)



#%%

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=( img_rows, img_cols,3)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(k))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')

#%%

hist = model.fit(X_train, Y_train, batch_size=24, nb_epoch=nb_epoch,
               verbose=1, validation_data=(X_test, Y_test))              
              
              
              
fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)              
              