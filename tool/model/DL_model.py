#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

tf.random.set_seed(2)

def dnn(input_shape, label):
    model_input = Input(shape=(input_shape,))
    m = Dense(128, activation='relu')(model_input)
    m = Dropout(0.3)(m)
    m = Dense(256, activation='relu')(m)
    m = Dropout(0.3)(m)
    m = Dense(512, activation='relu')(m)
    m = Dropout(0.3)(m)
    m = Dense(512, activation='relu')(m)
    m = Dropout(0.3)(m)
    m = Dense(256, activation='relu')(m)
    m = Dropout(0.3)(m)
    m = Dense(128, activation='relu')(m)
    m = Dropout(0.3)(m)
    m = Dense(64, activation='relu')(m)
    m = Dropout(0.3)(m)
    if label == 1:
        model_output = Dense(label, activation='sigmoid')(m)
    else:
        model_output = Dense(label, activation='sofmax')(m)
    
    model = Model(model_input, model_output)

    model.model_name = "DNN"
    
    return model

