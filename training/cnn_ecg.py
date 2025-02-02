# -*- coding: utf-8 -*-
"""CNN_ECG.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jEdsmJ0bDuxKjGW0nDHn4_vkOnIFDQW5
"""

#unzip "/content/data.zip"

#IMAGE PROCESSING

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

x_train = train_datagen.flow_from_directory("/data/tain",target_size = (64,64),batch_size = 32,class_mode = "categorical")
x_test = test_datagen.flow_from_directory("/data/test",target_size = (64,64),batch_size = 32,class_mode = "categorical")

x_train.class_indices

#MODEL BUILDING

model = Sequential()

model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = "relu"))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # ANN Input...

#Adding Dense Layers

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 128,kernel_initializer = "random_uniform",activation = "relu"))

model.add(Dense(units = 6,kernel_initializer = "random_uniform",activation = "softmax"))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(generator=x_train,steps_per_epoch = len(x_train), epochs=9, validation_data=x_test,validation_steps = len(x_test))

#Saving Model.
model.save('ECG.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('ECG.h5')

img=image.load_img("/content/Unknown_image.png",target_size=(64,64))

x=image.img_to_array(img)

import numpy as np

x=np.expand_dims(x,axis=0)

pred = model.predict(x)
y_pred=np.argmax(pred)
y_pred

index=['left Bundle Branch block',
       'Normal',
       'Premature Atrial Contraction',
       'Premature Ventricular Contraction',
       'Right Bundle Branch Block',
       'Ventricular Fibrillation']
result = str(index[y_pred])
#return result

