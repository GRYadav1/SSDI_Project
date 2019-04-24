#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle

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
from encoder1 import encoder
from data_preprocess import process_image

#get_ipython().run_line_magic('run', 'data_preprocess.ipynb')
#get_ipython().run_line_magic('run', 'encoder.ipynb')
print("Compiling predict...")

# In[62]:


def prepare_pred_data(im1,im2):
    prepared_data = []
    print(im1)
    print(im2)
    #for i in range(X.shape[0]):
    input_im1 = process_image(im1)
    input_im2 = process_image(im2)
    print("Input_Im1:",input_im1.shape)
    encoded_img1 = encoder.predict(input_im1)
    encoded_img2 = encoder.predict(input_im2)
    print(encoded_img1.shape)

    encoded_img1 = encoded_img1.flatten()
    encoded_img2 = encoded_img2.flatten()

    '''mean1 = np.mean(encoded_img1 - encoded_img2)
    mean2 = np.mean(encoded_img1 + encoded_img2)
    euclidean_distance = np.sum((encoded_img1-encoded_img2)**2)
    im_sum = np.sum((encoded_img1-encoded_img2))

    data = [mean1,mean2,euclidean_distance,im_sum]
    prepared_data.append(data)'''

    result_im = ((encoded_img1-encoded_img2)**2)
    prepared_data.append(result_im)

    return prepared_data


# In[63]:


def predict_result(im1, im2):

    test_prepared_data = prepare_pred_data(im1,im2)
    test_prepared_data = np.array(test_prepared_data)
    test_prepared_data = test_prepared_data/np.mean(test_prepared_data)
    loaded_model = tf.keras.models.load_model('checkpoint/trained_model.h5')
    class_names = ['Genuine','Forged']
    test_predictions = loaded_model.predict(test_prepared_data)
    print(test_predictions)
    result = class_names[np.argmax(test_predictions[0] )]
    print(result )

    return result


# In[65]:


##predict_result( 'data/test/066/03_066.png', 'data/test/066/11_066.png')
predict_result( 'data/test/066/10_066.png', 'data/test/066/11_066.png')


# In[ ]:
