#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Lambda, merge, Dense, Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model, Sequential, model_from_json, load_model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
#import torch
import data_preprocess
#get_ipython().run_line_magic('run', 'data_preprocess.ipynb')


# In[9]:


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
#x = Flatten()(x)
encoded = Dense(256, activation='relu')(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder = Model(input_img, encoded)


# In[10]:




def prepare_data(X):
    prepared_data = []

    for i in range(X.shape[0]):
        input_im1 = process_image(X[i,0])
        input_im2 = process_image(X[i,1])
        
        encoded_img1 = encoder.predict(input_im1)
        encoded_img2 = encoder.predict(input_im2)

        encoded_img1 = encoded_img1.flatten()
        encoded_img2 = encoded_img2.flatten()
        
        '''mean1 = np.mean(encoded_img1 - encoded_img2)
        mean2 = np.mean(encoded_img1 + encoded_img2)
        euclidean_distance = np.sum((encoded_img1-encoded_img2)**2)
        im_sum = np.sum((encoded_img1-encoded_img2))'''

        #data = [mean1,mean2,euclidean_distance,im_sum]
        result_im = ((encoded_img1-encoded_img2)**2)
        prepared_data.append(result_im)
        #prepared_data.append(data)
    
    return prepared_data


# In[ ]:




