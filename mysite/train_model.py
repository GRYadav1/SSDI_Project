#!/usr/bin/env python
# coding: utf-8

# In[18]:


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
#%run encoder.ipynb


# In[19]:


train_prepared_data = prepare_data(X)
train_prepared_data = np.array(train_prepared_data)
input_x  = train_prepared_data/np.mean(train_prepared_data)


# In[20]:


class_names = ['Genuine','Forged']


# In[21]:


clf = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])


# In[22]:


clf.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              #  loss = 'binary_crossentropy',
              metrics=['accuracy'])


# In[24]:


#clf.fit(input_x, T, epochs=500, batch_size=50)
clf.fit(input_x, T, epochs=100)


# In[25]:


#
test_data = pd.read_csv("data/validation_data.csv")
test_data = test_data.iloc[np.random.permutation(len(test_data))]
#test_data = test_data[:20]
test_X = test_data.iloc[:,:-1].values
test_T = test_data.iloc[:,-1].values

test_X[:,0] = 'data/train/'+test_X[:,0]
test_X[:,1] = 'data/train/'+test_X[:,1]

test_prepared_data = prepare_data(test_X)

test_prepared_data = np.array(test_prepared_data)
test_prepared_data  = test_prepared_data/np.mean(test_prepared_data)

test_loss, test_acc = clf.evaluate(test_prepared_data, test_T)
print('Test accuracy:', test_acc)


# In[26]:


test_predictions = clf.predict(test_prepared_data)
print(test_T)

prediction = []
for i in range(test_predictions.shape[0]):
    prediction.append(np.argmax(test_predictions[i]))
    
print(np.array(prediction) )


# In[27]:


# serialize model to JSON
model_json = clf.to_json()
with open("checkpoint/model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
clf.save_weights("checkpoint/model.h5")
print("Saved model to disk")


# In[28]:


clf.save('checkpoint/trained_model.h5')  # creates a HDF5 file 'my_model.h5'
print("Saved model to disk")

# returns a compiled model
# identical to the previous one


# In[ ]:




