# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:49:14 2017

@author: DELL PC
"""

# -*- coding: utf-8 -*-


import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.__version__

PATH = os.getcwd()

LOG_DIR = PATH+ '/logs'
#metadata = os.path.join(LOG_DIR, 'metadata2.tsv')

#%%
data_path = PATH + '//data'
data_dir_list = os.listdir(data_path)

img_data=[]
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        input_img_resize=cv2.resize(input_img,(32,32))
        img_data.append(input_img_resize)
    
                
img_data = np.array(img_data)

#%%

feature_vectors = np.loadtxt('feature_vectors_samples_testAug.txt')
print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])

num_of_samples=feature_vectors.shape[0]
#num_of_samples_each_class = 13

features = tf.Variable(feature_vectors, name='features')

PATH=os.getcwd()
path1 = '//data//'    #path of folder of images    
  #path of folder to save images    

listing = os.listdir(PATH+path1) 
img=[]
i=0
for folder in listing:
    print(folder)
    img.append(os.listdir(PATH+path1+folder+"//"))
    i+=1
 
#num_samples=i    
#label=np.ones((num_samples,),dtype = int)
#len(img[0])
#label[0:len(img[0])]=0
#l=len(img[0])
#label[len(img[0]):l+len(img[1])]=1
#l+=len(img[1])
#label[l:l+len(img[2])]=2
#l+=len(img[2])
#label[l:]=3
#    
#y = label
##%%
#dirname='data'
#length_dir=[]
#for class_directory in os.listdir(dirname):
#    length_dir.append(len(os.listdir(os.path.join(dirname,class_directory))))
#
#           
## 
#num_samples=0
#for i in range(len(length_dir)):
#    num_samples+=length_dir[i] 
#
#array=[]
#array.append(0)
#k=0
#for i in range(len(length_dir)):
#    k+=length_dir[i]
#    array.append(k)
#           
#label=np.ones((num_samples,),dtype = int)
#
#for i in range(len(array)-1):
#    label[array[i]:array[i+1]]=i
#    
#k= len(array)-1
#label[array[k]:]=k 

#%%
names =listing



'''' creating metadata .tsv'''

metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
metadata_file.write('Class\tName\n')
#k=num_of_samples_each_class
j=0
#for i in range(210):
#    metadata_file.write('%06d\t%s\n' % (i, names[y[i]]))
for i in range(len(img)):
    for j in range(len(img[i])):
        c = names[i]
        metadata_file.write('{}\t{}\n'.format(i,c))

metadata_file.close()
       


#%%


    
# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

sprite = images_to_sprite(img_data)
cv2.imwrite(os.path.join(LOG_DIR, 'sprite_4_classes.png'), sprite)
#scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

#%%
with tf.Session() as sess:
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images_4_classes.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_4_classes.tsv')
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite_4_classes.png')
    embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
