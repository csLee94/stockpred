import pandas as pd
import pandas_datareader as pdr

code_data = pd.read_csv("./kospi.csv", encoding='cp949')
target_dict={}
target_df = pd.DataFrame()
count = 0
for code in list(code_data['code']):
    s_code = code[3:-3]
    try:
        tdf = pdr.get_data_yahoo(s_code+".KS")
        target_dict['code'] = code
        target_dict['name'] = str(code_data.loc[code_data['code']==code]['name'].values).split("'")[1]
        target_df = target_df.append(target_dict, ignore_index=True)
        count +=1
        print(count)
    except:
        print("pass")

# target_df = pd.DataFrame(target_dict)
target_df.to_csv("./kospi_list.csv", encoding='cp949')