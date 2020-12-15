# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 23:31:38 2020

@author: Home
"""

import keras
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Input, Conv2D, Conv2DTranspose

from keras.layers import concatenate,Add,MaxPool2D
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from keras.models import Model


keras.__version__

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2, 
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train = train_datagen.flow_from_directory(r'C:\Users\Home\Desktop\spare\data\horse dataset\horse-or-human\horse-or-human\train',
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')
# class indices
train.class_indices

valid = ImageDataGenerator(rescale=1./255)

vali = valid.flow_from_directory(r'C:\Users\Home\Desktop\spare\data\horse dataset\horse-or-human\horse-or-human\validation',
                                  target_size=(150,150),
                                  batch_size=32,
                                  class_mode='binary')


type(train)


cnn = Sequential()
cnn.add(Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[150,150,3]))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(Dropout(0.5))

cnn.add(Flatten())
cnn.add(Dense(units=128,activation='relu'))
cnn.add(Dropout(0.1))

cnn.add(Dense(units=256,activation='relu'))
cnn.add(Dropout(0.25))

cnn.add(Dense(units=1,activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.summary()

path = r'C:\Users\Home\Desktop\spare\data\horse dataset\horse-or-human\k.h5'
checkpoint = ModelCheckpoint(path,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]


history = cnn.fit(train,
                  epochs=2,
                  verbose=1,
                  validation_data=vali,
                  callbacks=callbacks_list)


import numpy as np
from keras.preprocessing import image
from keras.models import load_model, Model


model1 = load_model(path)

#def pred(images):
#    test_image = image.load_img(r'C:\Users\Home\Desktop\spare\data\horse dataset\horse-or-human\horse.jpg',target_size=(150,150))
#    test_image = image.img_to_array(test_image)/255
#    test_image = np.expand_dims(test_image,axis=0)
#    
#    #result = model1.predict(test_image)
#    
#    result1 = model1.predict_classes(test_image)
#    
#    
#    if result1 == 0:
#        print('horse')
#    else:
#        print('human')



test_image = image.load_img(r'C:\Users\Home\Desktop\spare\data\horse dataset\horse-or-human\horse.jpg',target_size=(150,150))
test_image = image.img_to_array(test_image)/255
test_image = np.expand_dims(test_image,axis=0)

#result = model1.predict(test_image)

result1 = model1.predict_classes(test_image)


if result1 == 0:
    print('horse')
else:
    print('human')

