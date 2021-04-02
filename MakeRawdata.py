import numpy as np
import pandas as pd
import pandas_datareader as pdr
from matplotlib import pyplot as plt 
import datetime
import os
import time

# parameter
'''
1. code
2. refer_day
3. timestamp
'''
def makerawdata(code, refer_day, timestamp):
    # 주가 정보 불러오기
    start = datetime.datetime(2000,1,1)
    df = pdr.get_data_yahoo(str(code)+".KS", start)
    # 기초 변수 설정
    col_list = ['High', 'Low', 'Open', 'Close', 'Adj Close']
    col_dict = {'High': 'olive', 'Low':'forestgreen', 'Open':'darkmagenta', 'Close':'darkviolet', 'Adj Close':'crimson'}
    start_num = 0
    # 중복 검사를 위한 리스트 생성
    category_list = ["0", "1","2","3","4","5","6","7","8","9","10","11"]
    file_list =[]
    for dir_num in category_list:
        dir_title = str(refer_day)+str(timestamp)
        path_dir = "./img/%s/%s"
        file_list += os.listdir(path_dir % (dir_title, dir_num))
    # 그래프 생성 While 문
    while True:
        if len(df) < start_num + refer_day + timestamp:
            break
        tdf = df[start_num:start_num+refer_day]
        tdf_mean = np.array(tdf['Adj Close']).mean()
        tdf_std = np.array(tdf['Adj Close']).std()
        fig, ax1 = plt.subplots()
        x = 320 / fig.dpi
        y = 240 / fig.dpi
        fig.set_figwidth(x)
        fig.set_figheight(y)
        ax2 =  ax1.twinx()
        ax1.axis([tdf.index[0],tdf.index[refer_day-1],(tdf_mean-tdf_std*4), (tdf_mean+tdf_std*4)])
        ax2.axis([tdf.index[0],tdf.index[refer_day-1],(np.array(tdf['Volume']).mean()- 4*np.array(tdf['Volume']).std()),(np.array(tdf['Volume']).mean()+4*np.array(tdf['Volume']).std())])
        for col in col_list:
            ax1.plot(tdf[col], color = col_dict[col], linewidth= 3.0, alpha=0.6)
        ax2.plot(tdf['Volume'], color = 'gold', linewidth=3.0, alpha=0.6) # High, Low, Open, Close, Adj Close, Volume 그래프 완성
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        ax2.axes.yaxis.set_visible(False)
        ##################################
        e_price = tdf['Adj Close'][refer_day-1]
        p_price = df["Adj Close"][start_num+refer_day+timestamp-1]
        num = ((p_price-e_price)/e_price)*100
        target_ratio = round(num, 2)
        if target_ratio > 10:
            category = "0"
        elif target_ratio <= 10 and target_ratio > 8:
            category = "1"
        elif target_ratio <= 8 and target_ratio > 6:
            category = "2"
        elif target_ratio <= 6 and target_ratio > 4:
            category = "3"
        elif target_ratio <= 4 and target_ratio > 2:
            category = "4"
        elif target_ratio <= 2 and target_ratio > 0:
            category = "5"
        elif target_ratio <= 0 and target_ratio > -2:
            category = "6"
        elif target_ratio <= -2 and target_ratio > -4:
            category = "7"
        elif target_ratio <= -4 and target_ratio > -6:
            category = "8"
        elif target_ratio <= -6 and target_ratio > -8:
            category = "9"
        elif target_ratio <= -8 and target_ratio > -10:
            category = "10"
        elif target_ratio <= -10:
            category = "11"

        title = str(list(tdf.index)[0]).split(" ")[0].replace("-","_")
        result = str(target_ratio).replace(".", "^")
        if "%s_%s_%s" % (code, title, result) in file_list:
            pass
        else:
            fig.savefig("./img/%s/%s/%s_%s_%s.png" % (dir_title,category, code, title, result))
        start_num += 1
        plt.close(fig)
        



refer_day = 30
timestamp = 5

code_data = pd.read_csv("./kospi.csv")
for code in list(code_data['code']):
    s_code = code[4:-3]
    try:
        makerawdata(s_code, refer_day, timestamp)
        name = str(code_data.loc[code_data['code']==code]['name'])
        time_tuple = time.localtime()
        time_str = time.strftime("%m/%d/%Y, %H:%M:%S", time_tuple).split(", ")[0]
        memo = open('memo.txt', 'a', encoding='utf8')
        memo.write(str(name)+"   "+time_str+'\n')
        memo.close()
        print("Done: %s" % str(name))
        time.sleep(2)
    except:
        print("Something is wrong")
