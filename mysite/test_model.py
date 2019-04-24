#!/usr/bin/env python
# coding: utf-8

# In[24]:


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
import torch

get_ipython().run_line_magic('run', 'data_loader.ipynb')


# In[25]:


test_prepared_data = prepare_data(test_X)
test_prepared_data = np.array(test_prepared_data)
test_prepared_data  = test_prepared_data/np.mean(test_prepared_data)


# In[26]:


loaded_model = tf.keras.models.load_model('checkpoint/trained_model.h5')


# In[27]:


test_losses, test_accuracy = loaded_model.evaluate(test_prepared_data, test_T)
print('Test accuracy:', test_accuracy)


# In[28]:


test_predictions = loaded_model.predict(test_prepared_data)
#print(test_predictions)


# In[29]:


print(test_T)

prediction = []
for i in range(test_predictions.shape[0]):
    prediction.append(np.argmax(test_predictions[i]))
    
print(np.array(prediction) )


# In[ ]:




