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
print(model.summary())