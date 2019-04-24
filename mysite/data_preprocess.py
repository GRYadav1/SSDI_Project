#!/usr/bin/env python
# coding: utf-8

# In[5]:


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

print("Inside data_preprocess....");
# In[6]:


def process_image(img_name):
    img_list = list()

    im = cv2.imread(img_name)
    im = cv2.resize(im, (28,28))
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    img_list.append(im)
    img_list = np.asarray(img_list)
    img_list = np.reshape(img_list, (1, 28, 28, 1))
    return img_list
