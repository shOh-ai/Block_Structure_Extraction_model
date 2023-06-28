#!/usr/bin/env python
# coding: utf-8

# ## Google Mount

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## import Packages

# In[ ]:


import numpy as np
import os
import glob
import pandas as pd
import cv2
import io
import keras
import sklearn
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Dense, Activation
from keras.models import Sequential, load_model
from keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import IPython
from keras.utils import img_to_array


# ## csv 파일 및 이미지 데이터 불러오기

# In[ ]:


df_train = pd.read_csv('/content/drive/MyDrive/train_split.csv')
df_val = pd.read_csv('/content/drive/MyDrive/validation_split.csv')
df_test = pd.read_csv('/content/drive/MyDrive/test.csv')


# In[ ]:


# 이미지 폴더 경로
directory_train = "/content/drive/MyDrive/train_img_output"
directory_val = "/content/drive/MyDrive/validation_img_output"
directory_test = "/content/drive/MyDrive/test_img_output"

x_col = "img_path"


# In[ ]:


def get_train_generator(df_train, directory_train, x_col, columns, shuffle=True, batch_size=256, seed=30, target_w = 224, target_h = 224):
    # normalize images
    image_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)

    generator = image_generator.flow_from_dataframe(
            dataframe = df_train,
            directory = directory_train,
            x_col = x_col,
            y_col = columns,
            class_mode="raw",
            shuffle = shuffle,
            batch_size = batch_size,
            seed = seed,
            target_size=(target_w,target_h))
    
    return generator

def get_validation_generator(df_val, directory_val, x_col, columns, shuffle = False, batch_size=128, seed=30, target_w = 224, target_h = 224):
    # normalize images
    image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True)

    # get validation generator
    validation_generator = image_generator.flow_from_dataframe(
        dataframe = df_val,
        directory = directory_val,
        x_col = x_col,
        y_col = columns,
        class_mode = "raw",
        shuffle = shuffle,
        batch_size = batch_size,
        seed = seed,
        target_size=(target_w,target_h))
    
    return validation_generator

def get_test_generator(df_test, directory_test, x_col, columns, shuffle = False, batch_size=128, seed=30, target_w = 224, target_h = 224):
    # normalize images
    image_generator = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True)

    # get test generator
    test_generator = image_generator.flow_from_dataframe(
        dataframe = df_test,
        directory = directory_test,
        x_col = x_col,
        y_col = columns,
        class_mode = "raw",
        shuffle = shuffle,
        batch_size = batch_size,
        seed = seed,
        target_size=(target_w,target_h))
    
    return test_generator


# In[ ]:


train_generator = get_train_generator(df_train, directory_train, "img_path", columns)
validation_generator = get_validation_generator(df_val, directory_val, "img_path", columns)
test_generator = get_test_generator(df_test, directory_test, "img_path", columns)


# In[ ]:


steps_per_epoch = train_generator.samples//train_generator.batch_size
steps_per_epoch


# In[ ]:


validation_steps = validation_generator.samples//validation_generator.batch_size
validation_steps


# In[ ]:


test_steps = test_generator.samples//test_generator.batch_size
test_steps


# ## MobileNetV2 model building

# In[ ]:


# 모델 불러오기
import tensorflow_hub as hub

mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" #사전 훈련된 mobilenetV2 다운
mobile_net_layers = hub.KerasLayer(mobilenet_v2, input_shape=(224,224,3)) #사전 훈련된 MobilenetV2 레이어 추출, 빈 이미지 크기 지정
mobile_net_layers.trainable = False #재교육 안 하도록

# 모델 구축
n_outputs = len(columns)

model = Sequential([
    mobile_net_layers,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_outputs, activation='sigmoid')]) #TensorFlow를 사용하여 사용자 정의 레이어 추가
  
model.summary() #모델 아키텍쳐 확인


# In[ ]:


from keras.callbacks import LearningRateScheduler

def step_decay(epoch):
    start = 0.01
    drop = 0.4
    epochs_drop = 5.0
    lr = start * (drop ** np.floor((epoch)/epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay, verbose=1)


# In[ ]:


# 모델 컴파일
model.compile(optimizer='adam',
              loss= 'binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


# test 모델 피팅
history = model.fit(train_generator,
                    validation_data = test_generator,
                    epochs = 40,
                    callbacks=[lr_scheduler],
                    steps_per_epoch = 50,
                    validation_steps = 11)


# ## MobileNetV2의 submission 파일 만들기

# In[ ]:


y_pred = model.predict(test_generator, steps = len(test_generator))
                                         
type(y_pred)
print(y_pred.shape)
print(y_pred.ndim)
print(y_pred.size)
print(y_pred)


# In[ ]:


def predict(y_df):
    preds=[]
    for i in y_df:
        for j in range(10):
            if i[j] > 0.5:
                preds.append(1)
            else:
                preds.append(0)
    return np.array(preds)

pred_arr = predict(y_pred)

print(pred_arr.shape)
print(pred_arr.ndim)
print(pred_arr.size)
print(pred_arr)

np_resha = pred_arr.reshape(1460,10)
pred_y = np_resha
print(pred_y)


# In[ ]:


submit = pd.read_csv('/content/drive/MyDrive/sample_submission.csv')


# In[ ]:


submit.iloc[:, 1:] = pred_y  ### 여기서 pres = [[0, 1, 1, 0, 0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1, 1, 0, 0], ..., [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]]
submit.head()


# In[ ]:


submit.to_csv('submit_MobileNetV2.csv', index=False)

