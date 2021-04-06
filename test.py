from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.utils import np_utils

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt

print(time.strftime('%X', time.localtime(time.time())))

category_list = ["0", "1","2","3","4","5","6","7","8","9","10","11"]

file_dict ={}
for label in category_list:
    path_dir = "C:/Users/pc1/Documents/#0.LCS/stockpred/img/305/%s"
    file_dict[label] = os.listdir(path_dir % label)
    
x_data = []
y_data = []
path = "C:/Users/pc1/Documents/#0.LCS/stockpred/img/305/%s/%s"

for label in category_list:
    file_list = file_dict[label]
    for infor in file_list[:1000]: #1000
        img = cv2.imread(path % (label, infor))
        x_data.append(img/255)
        y_data.append(label)
    time.sleep(0.1)

# print(time.strftime('%X', time.localtime(time.time())))

np_x_data = np.array(x_data)
np_y_data = np.array(y_data)
np_y_encoding = np_utils.to_categorical(np_y_data)

np_x_train, np_x_test, np_y_train, np_y_test = train_test_split(np_x_data, np_y_encoding, test_size=0.2)

model = Sequential()
model.add(Conv2D(12, kernel_size=(5,5), activation='relu', input_shape=(120,160, 3), strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu', strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(20, kernel_size=(3,3), activation='relu', strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(np_x_train, np_y_train, batch_size=25, epochs = 300)
model.save("./stockpred_cnn")

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()
loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

fig.savefig("./result.png")
plt.close()

print(time.strftime('%X', time.localtime(time.time())))

