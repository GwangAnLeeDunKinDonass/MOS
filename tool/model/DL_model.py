#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

def dnn(input_shape):
    model_input = Input(shape=(input_shape,))
    m = Dense(128, activation='relu')(model_input)
    m = Dense(64, activation='relu')(model_input)
    m = Dropout(0.3)(m)
    m = Dense(32, activation='relu')(model_input)
    model_output = Dense(1, activation='sigmoid')(m)
    
    model = Model(model_input, model_output)

    model.model_name = "DNN"
    
    return model

