import pandas as pd
import pandas_datareader as pdr
from matplotlib import pyplot as plt 
import datetime
code_df = pd.read_csv('./kospi200_list.csv')

start = datetime.datetime(2021,2,1)
df = pdr.get_data_yahoo("352820.KS", start)

print(df.iloc[30])
print("#"*30)
refer_day = 30
timestamp = 15

for idx in range(refer_day,len(df)):
    print(df.iloc[idx])