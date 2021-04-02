import os
import shutil
import random

category_list = ["1","2","3","4","5","6","7","8","9","10","11","12"]
path = "./img/305/%s"
copy_path = "./img/copy/%s/%s"
file_nm = ""

file_list =[]
for dir_num in category_list:
    file_list = os.listdir(path % dir_num)
    for idx in range(100):
        temptitle = random.choice(file_list)
        temppath = path + "/%s"
        shutil.copyfile(temppath % (dir_num, temptitle), copy_path % (dir_num, temptitle))
