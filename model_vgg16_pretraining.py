import tensorflow as tf

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense, Input, Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import regularizers

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
import random 

model = VGG16(weights=None, include_top=False, input_tensor=Input(shape=(120, 160, 3)))

additional_model = Sequential()
additional_model.add(model)
additional_model.add(Flatten())
additional_model.add(Dense(2048, kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),activation='relu'))
additional_model.add(Dropout(0.5))
additional_model.add(Dense(1024, kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),activation='relu'))
additional_model.add(Dropout(0.5))
additional_model.add(Dense(512, kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001),activation='relu'))
additional_model.add(Dropout(0.5))
additional_model.add(Dense(12, activation='softmax'))
additional_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

category_list = ["0", "1","2","3","4","5","6","7","8","9","10","11"]
x_data = []
y_data = [] 

path_dir = "D:/MyDocument/Desktop/temp_lcs/stockpred/img/305/%s"
path = "D:/MyDocument/Desktop/temp_lcs/stockpred/img/305/%s/%s"
# save_path = "./%s" # PC1
save_path = "D:/MyDocument/Desktop/temp_lcs/stockpred/%s" # PC2

for label in category_list:
    file_list = os.listdir(path_dir % label)    
    for infor in file_list[:500]: #1000
        img = cv2.imread(path % (label, infor))
        img = cv2.resize(img, dsize=(160,120), interpolation=cv2.INTER_AREA)
        x_data.append(np.array(img/255))
        y_data.append(int(label))
    time.sleep(0.1)

np_x_data = np.array(x_data)
np_y_data = np.array(y_data)
np_y_encoding = np_utils.to_categorical(np_y_data)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=50)
hist = additional_model.fit(np_x_data, np_y_encoding, batch_size=32, epochs = 100, validation_split=0.1, callbacks=[es]) # epochs=500 
additional_model.save(save_path % "vgg16")

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='validation loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='validation acc')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

fig.savefig("./result/vgg16_%s.png" % idx)
plt.close()