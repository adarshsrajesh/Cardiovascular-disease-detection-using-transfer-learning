import tensorflow as tf

from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,ZeroPadding2D,Conv2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator,DirectoryIterator
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import os 
import numpy as np
import shutil
import random


train_dir='data\train'
test_dir='data\test'

train_generator=ImageDataGenerator(rotation_range=20,
                                   rescale=1./255,)

classes=["LBblock","Normal","Pacontraction","pvcontraction","rbblock","vf"]

train_set = DirectoryIterator(train_dir,train_generator,target_size=(224,224),color_mode='grayscale',batch_size=16,classes=classes,class_mode='categorical')

test_generator=ImageDataGenerator(rescale=1./255)

test_set = DirectoryIterator(test_dir,test_generator,target_size=(224,224),color_mode='grayscale',batch_size=16,classes=classes,class_mode='categorical')