import pandas as pd
import pandas_datareader as pdr

df = pd.read_csv('kospi200_list.csv', encoding = 'cp949')

fail_lst = []
for idx in range(len(df)): # len(df)
    txt = df.iloc[idx]['name']
    code= df.iloc[idx]['code']
    code= code.split('A')[-1]
    try:
        pdr.get_data_yahoo(str(code)+".KS")
    except:
        fail_lst.append(txt)

print(fail_lst)