import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
import random 
# model = tf.keras.models.load_model('stockpred_cnn')

category_list = ["0", "1","2","3","4","5","6","7","8","9","10","11"]


file_dict ={}
file_list = []
for label in category_list:
    path_dir = "./img/305/%s"
    file_list += os.listdir(path_dir % label)
    random.shuffle(file_list)


path = "D:/MyDocument/Desktop/temp_lcs/stockpred/img/305/%s/%s"
save_path = "D:/MyDocument/Desktop/temp_lcs/stockpred/%s"
idx = 0
while True:
    x_data = []
    y_data = []
    if (idx+1)*10000 > len(file_list):
        for infor in file_list[idx*10000:]: # 범위 지정
            target_ratio = infor.split("_")[-1].rstrip(".png")
            target_ratio = float(target_ratio.replace("^", "."))
            if target_ratio > 10:
                category = "0"
            elif target_ratio <= 10 and target_ratio > 8:
                category = "1"
            elif target_ratio <= 8 and target_ratio > 6:
                category = "2"
            elif target_ratio <= 6 and target_ratio > 4:
                category = "3"
            elif target_ratio <= 4 and target_ratio > 2:
                category = "4"
            elif target_ratio <= 2 and target_ratio > 0:
                category = "5"
            elif target_ratio <= 0 and target_ratio > -2:
                category = "6"
            elif target_ratio <= -2 and target_ratio > -4:
                category = "7"
            elif target_ratio <= -4 and target_ratio > -6:
                category = "8"
            elif target_ratio <= -6 and target_ratio > -8:
                category = "9"
            elif target_ratio <= -8 and target_ratio > -10:
                category = "10"
            elif target_ratio <= -10:
                category = "11"
            img = cv2.imread(path % (category, infor))
            img = cv2.resize(img, dsize=(160,120), interpolation=cv2.INTER_AREA)
            x_data.append(np.array(img/255))
            y_data.append(category)
        np_x_data = np.array(x_data)
        np_y_data = np.array(y_data)    
        np_y_encoding = np_utils.to_categorical(np_y_data) 
        model = tf.keras.models.load_model('stockpred_cnn')
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=300)
        hist = model.fit(np_x_data, np_y_encoding, batch_size=50, epochs = 500, validation_split=0.2, callbacks=[es]) # epochs=500 
        model.save(save_path % "stockpred_cnn")

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

        fig.savefig("./result_%s.png" % idx)
        plt.close()
        break

    else:
        for infor in file_list[idx*10000:(idx+1)*10000]: # 범위 지정
            target_ratio = infor.split("_")[-1].rstrip(".png")
            target_ratio = float(target_ratio.replace("^", "."))
            if target_ratio > 10:
                category = "0"
            elif target_ratio <= 10 and target_ratio > 8:
                category = "1"
            elif target_ratio <= 8 and target_ratio > 6:
                category = "2"
            elif target_ratio <= 6 and target_ratio > 4:
                category = "3"
            elif target_ratio <= 4 and target_ratio > 2:
                category = "4"
            elif target_ratio <= 2 and target_ratio > 0:
                category = "5"
            elif target_ratio <= 0 and target_ratio > -2:
                category = "6"
            elif target_ratio <= -2 and target_ratio > -4:
                category = "7"
            elif target_ratio <= -4 and target_ratio > -6:
                category = "8"
            elif target_ratio <= -6 and target_ratio > -8:
                category = "9"
            elif target_ratio <= -8 and target_ratio > -10:
                category = "10"
            elif target_ratio <= -10:
                category = "11"
            img = cv2.imread(path % (category, infor))
            img = cv2.resize(img, dsize=(160,120), interpolation=cv2.INTER_AREA)
            x_data.append(np.array(img/255))
            y_data.append(category)

        np_x_data = np.array(x_data)
        np_y_data = np.array(y_data)    
        np_y_encoding = np_utils.to_categorical(np_y_data) 
        model = tf.keras.models.load_model('stockpred_cnn')
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=300)
        hist = model.fit(np_x_data, np_y_encoding, batch_size=50, epochs = 500, validation_split=0.2, callbacks=[es]) # epochs=500 
        model.save(save_path % "stockpred_cnn")

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

        fig.savefig("./result_%s.png" % idx)
        plt.close()
        idx +=1 