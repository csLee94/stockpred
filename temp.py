# import pandas as pd
# import pandas_datareader as pdr
# from MakeRawdata import makerawdata
# import time

# code_data = pd.read_csv("./kospi_list.csv", encoding='cp949')
# print(list(code_data['code'][:3]))

# refer_day = 30
# timestamp = 5


# for code in list(code_data['code'][:3]):
#     s_code = code[3:-3]
#     makerawdata(s_code, refer_day, timestamp)
#     name = str(code_data.loc[code_data['code']==code]['name'].values).split("'")[1]
#     time_tuple = time.localtime()
#     time_str = time.strftime("%m/%d/%Y, %H:%M:%S", time_tuple).split(", ")[0]
#     memo = open('memo.txt', 'a', encoding='utf8')
#     memo.write(str(name)+"   "+time_str+'\n')
#     memo.close()
#     print("Done: %s" % str(name))

import tensorflow as tf
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

model = tf.keras.models.load_model('stockpred_cnn')

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
    for infor in file_list[1000:1010]: #1000
        img = cv2.imread(path % (label, infor))
        x_data.append(img/255)
        y_data.append(label)
    time.sleep(0.1)

np_x_data = np.array(x_data)
np_y_data = np.array(y_data)
np_y_encoding = np_utils.to_categorical(np_y_data)

pred = model.predict_classes(np_x_data)
print(pred)
print(np_y_data)
result = confusion_matrix(np_y_data.astype(str), pred.astype(str))
print(result)
result_df  = pd.DataFrame(result)
result_df.to_excel('./result.xlsx')