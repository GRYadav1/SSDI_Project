#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

get_ipython().run_line_magic('run', 'data_preprocess.ipynb')
get_ipython().run_line_magic('run', 'encoder.ipynb')


# In[12]:


data = pd.read_csv("data/train_data.csv")
data = data.iloc[np.random.permutation(len(data))]
#data = data[:200]
X = data.iloc[:,:-1].values

T = data.iloc[:,-1].values
X[:,0] = 'data/train/'+X[:,0]
X[:,1] = 'data/train/'+X[:,1]

# test data
test_data = pd.read_csv("data/test_data.csv")
#test_data = pd.read_csv("data/validation_data.csv")
test_data = test_data.iloc[np.random.permutation(len(test_data))]
#test_data = test_data[:20]
test_X = test_data.iloc[:,:-1].values
test_T = test_data.iloc[:,-1].values

test_X[:,0] = 'data/test/'+test_X[:,0]
test_X[:,1] = 'data/test/'+test_X[:,1]


# In[ ]:





# In[ ]:





# In[ ]:




