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

category_list = ["0", "1","2","3","4","5","6","7","8","9","10","11"]

x_data = []
y_data = []
path_dir = "D:/templcs/stockpred/img/305/%s"
path = "D:/templcs/stockpred/img/305/%s/%s"
save_path = "./%s" # PC1
save_path = "D:/MyDocument/Desktop/temp_lcs/stockpred/%s" # PC2

for label in category_list:
    file_list = os.listdir(path_dir % label)    
    for infor in file_list[:1000]: #1000
        img = cv2.imread(path % (label, infor))
        img = cv2.resize(img, dsize=(160,120), interpolation=cv2.INTER_AREA)
        x_data.append(np.array(img/255))
        y_data.append(label)
    time.sleep(0.1)

np_x_data = np.array(x_data)
np_y_data = np.array(y_data)
np_y_encoding = np_utils.to_categorical(np_y_data)

model = Sequential()
model.add(Conv2D(12, kernel_size=(5,5), activation='relu', input_shape=(120, 160, 3), strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu', strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(20, kernel_size=(3,3), activation='relu', strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(160, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

fig.savefig(save_path % "result.png")
plt.close()

print(time.strftime('%X', time.localtime(time.time())))
 

