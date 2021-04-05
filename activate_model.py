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

refer_day =30
timestamp=5
category_list = ["0", "1","2","3","4","5","6","7","8","9","10","11"]

file_dict ={}
for label in category_list:
    path_dir = "C:/Users/pc1/Documents/#0.LCS/stockpred/img/305/%s"
    file_dict[label] = os.listdir(path_dir % label)
    
x_data = []
y_data = []
path = "C:/Users/pc1/Documents/#0.LCS/stockpred/img/305/%s/%s"

for label in category_list:
    tlst = file_dict[label]
    for infor in file_list[:10000]:
        img = cv2.imread(path % (label, infor))
        x_data.append(img/255)
        y_data.append(label)
    time.sleep(1)

np_x_data = np.array(x_data)
np_y_data = np.array(y_data)
np_y_encoding = np_utils.to_categorical(np_y_data)

np_x_train = np_x_data[:9000]
np_y_train = np_y_encoding[:9000]
np_x_test = np_x_data[-1000:]
np_y_test = np_y_encoding[-1000:]

model = Sequential()
model.add(Conv2D(12, kernel_size=(5,5), activation='relu', input_shape=(240,320, 3), strides=(2,2), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu', strides=(2,2),padding='same'))
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
model.fit(np_x_data_train, np_y_train, batch=25, epochs = 1000)

pred = model.predict_classes(np_x_test)

real_dict ={
    "[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":0,
    "[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":1,
    "[0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]":2,
    "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]":3,
    "[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]":4,
    "[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]":5,
    "[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]":6,
    "[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]":7,
    "[0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]":8,
    "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]":9,
    "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]":10,
    "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]":11,

}
real_lst =[]
for infor in np_y_data_test:
  real_lst.append(real_dict[str(infor)])

real_np = np.array(real_lst)
result_df  = pd.DataFrame(coufusion_matrix(real_np, pred))
result_df.to_excel('./result.xlsx')
