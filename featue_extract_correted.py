# -*- coding: utf-8 -*-
"""
Created on Wed Aug 02 09:30:15 2017

@author: DELL PC
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:43:00 2017

@author: anas
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


import cv2

# input image dimensions
img_rows, img_cols = 32, 32



dirname='data'


# number of channels
img_channels = 3





#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 100


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

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
model.add(Dense(len(os.listdir(dirname))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')


#%%

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)

model.layers.pop() # Get rid of the classification layer
model.layers.pop() # Get rid of the dropout layer
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []


#%%

dim=(img_rows, img_cols)
image_list=[]
for class_directory in os.listdir(dirname): 
    if os.path.isdir(os.path.join(dirname, class_directory)): 
        for filename in os.listdir(os.path.join(dirname,class_directory)): 
            
            img_path = os.path.join(dirname, class_directory, filename)
            if img_path.endswith(".jpg"): 
                #print("Loading Image")
                #img=cv2.imread(img_path)
                img = Image.open(img_path).resize((dim))
                img = np.array(img)
                #normalize_image = readandnormalizeImage(img, dim,img_path)
                #This is for visualising the input images #
                #save_path="C:\\Visualisation\\testImage\\"
                #write_path=os.path.join(save_path+filename)
                #write_image=normalize_image.reshape(64,64,3).astype(np.uint8)
                #io.imsave(write_path,write_image)
                # This might not me needed while doing training #
                imgg=img.reshape(1,img_rows,img_cols,3)
                imgg = imgg.astype('float32')
                imgg /= 255
                image_list.append(imgg)
#%% 
image_features_list=[]        
for i in range(len(image_list)):
    image_features=model.predict(image_list[i])
    image_features_list.append(image_features)
    
image_features_arr=np.asarray(image_features_list)
image_features_arr = np.rollaxis(image_features_arr,1,0)
image_features_arr = image_features_arr[0,:,:]

np.savetxt('feature_vectors_samples_testAug.txt',image_features_arr)    
