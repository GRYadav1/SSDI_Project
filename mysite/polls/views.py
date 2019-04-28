from django.shortcuts import render

from django.http import HttpResponse
from django.template import loader
from sklearn_nn import train_model,test_Model
from sklearn_nn_predict import predict_test_data
import os
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

pathOrig ="data/test/050/"
pathForg= "data/test/050_forg/"

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

print("First imput_image:",input_img.shape)

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

def process_image_data(img_name):
    img_list = list()

    im = cv2.imread(img_name)
    im = cv2.resize(im, (28,28))
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    img_list.append(im)
    img_list = np.asarray(img_list)
    img_list = np.reshape(img_list, (1, 28, 28, 1))
    return img_list

def prepare_pred_data(im1,im2):
    prepared_data = []
    print(im1)
    print(im2)
    #for i in range(X.shape[0]):
    input_im1 = process_image_data(im1)
    input_im2 = process_image_data(im2)
    #print("Input_Im1:",input_im1.shape)
    encoded_img1 = encoder.predict(input_im1)
    encoded_img2 = encoder.predict(input_im2)
    #print(encoded_img1.shape)

    encoded_img1 = encoded_img1.flatten()
    encoded_img2 = encoded_img2.flatten()

    result_im = ((encoded_img1-encoded_img2)**2)
    prepared_data.append(result_im)

    return prepared_data

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


predict_result( pathOrig+'01_050.png', pathForg+'02_0126050.png')

def index(request):
    print("Before Call!!!")
    template = loader.get_template('polls/index.html')
    print("Hi index")
    #global clf
    #clf = train_model();
    return HttpResponse(template.render({}, request))

def imageSubmit(request):
    print("Hi imageSubmit")
    if 'q' in request.GET:
        pathString1=os.path.join(os.path.abspath(pathOrig),request.GET['p'])
        print("-------->"+request.GET['p'])
        pathString2= os.path.join(os.path.abspath(pathForg),request.GET['q'])
        #print("Print Paths:\n",pathString1,pathString2);
        #global clf
        #message=test_Model(pathString1,pathString2,clf)

        message = predict_result(pathString1,pathString2)
        #message=predict_test_data(pathString1,pathString2)
        print("Final Score :",message)

        #pathString+=request.GET['q']
    else:
        message = 'You know nothing, Jon Snow!'
        print(message)
    return HttpResponse(message)
