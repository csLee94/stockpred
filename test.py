import os



category_list = ["1","2","3","4","5","6","7","8","9","10","11","12"]
file_list =[]
for dir_num in category_list:
    dir_title = str(30)+str(15)
    path_dir = "./img/%s/%s"
    file_list += os.listdir(path_dir % (dir_title, dir_num))

tlst = ["img/123", "img/321", "iimg/123"]
if "img/" in tlst:
    print('yes')

