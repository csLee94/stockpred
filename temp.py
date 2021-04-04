import pandas as pd
import pandas_datareader as pdr
from MakeRawdata import makerawdata
import time

code_data = pd.read_csv("./kospi_list.csv", encoding='cp949')
print(list(code_data['code'][:3]))

refer_day = 30
timestamp = 5


for code in list(code_data['code'][:3]):
    s_code = code[3:-3]
    makerawdata(s_code, refer_day, timestamp)
    name = str(code_data.loc[code_data['code']==code]['name'].values).split("'")[1]
    time_tuple = time.localtime()
    time_str = time.strftime("%m/%d/%Y, %H:%M:%S", time_tuple).split(", ")[0]
    memo = open('memo.txt', 'a', encoding='utf8')
    memo.write(str(name)+"   "+time_str+'\n')
    memo.close()
    print("Done: %s" % str(name))
