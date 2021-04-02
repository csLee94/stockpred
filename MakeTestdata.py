import numpy as np
import pandas as pd
import pandas_datareader as pdr
from matplotlib import pyplot as plt 
import datetime
import os
import time


def maketestdata(code,refer_day,timestamp):
    start = datetime.datetime()
    df = pdr.get_data_yahoo(str(code)+".KS", start)
    # 기초 변수 설정
    col_list = ['High', 'Low', 'Open', 'Close', 'Adj Close']
    col_dict = {'High': 'olive', 'Low':'forestgreen', 'Open':'darkmagenta', 'Close':'darkviolet', 'Adj Close':'crimson'}
    start_num = 0
    