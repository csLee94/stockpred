import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time 

#-- 크롬드라이버 설정 
chrome_option = Options() #초기화
driver  = webdriver.Chrome('C:/Users/pc1/Documents/#0.LCS/project_job/job_ability/mysite/job/chromedriver.exe', chrome_options= chrome_option) # driver 선언
driver.get('https://finance.daum.net/domestic/market_cap')
page_path = "/html/body/div/div[4]/div[2]/div[1]/div[2]/div[2]/div/div/a[%s]"
xpath = '/html/body/div/div[4]/div[2]/div[1]/div[2]/div[2]/div/table/tbody/tr[%s]/td[2]/a'
## test 
# time.sleep(3)
# txt = driver.find_element_by_xpath(xpath % "1").text
# code = driver.find_element_by_xpath(xpath % "1").get_attribute('href')
# code = code.split('/')[-1]
# # buttom = driver.find_element_by_xpath(page_path % "1")
# # buttom.click()
# print(txt)
# print(code)

df = pd.DataFrame()
for idx in range(1,6):
    time.sleep(2)
    for nm in range(1,31):
        txt = driver.find_element_by_xpath(xpath % str(nm)).text
        code = driver.find_element_by_xpath(xpath % str(nm)).get_attribute('href')
        code = code.split('/')[-1]
        tdf = pd.DataFrame([[code, txt]], columns=['code', 'name'])
        df = df.append(tdf)
    buttom = driver.find_element_by_xpath(page_path % str(idx))
    buttom.click()

df.to_csv('./kospi200_list.csv', encoding='cp949')

