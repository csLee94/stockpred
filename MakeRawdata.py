import pandas as pd
import pandas_datareader as pdr
from matplotlib import pyplot as plt 
import datetime
code_df = pd.read_csv('./kospi200_list.csv')

start = datetime.datetime(2010,1,1)

df = pdr.get_data_yahoo("352820.KS", start)
print(df.iloc[-1])