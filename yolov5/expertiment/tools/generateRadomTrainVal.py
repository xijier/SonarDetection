import os
import shutil
import random

#file = "2007_val.txt"

#path_S = "E:\kg\data\SONAR_VOC_1/VOC2007/JPEGImages/000005.jpg"

path_val = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\VOC2007\ImageSets\Main/val.txt"
path_train = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\VOC2007\ImageSets\Main/train.txt"
path_test = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\VOC2007\ImageSets\Main/test.txt"

#shutil.copy(path_S,path_T)

base_folder = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\VOC2007/"
img_path = base_folder + "JPEGImages/"
image_files = os.listdir(img_path)
val_set_account = int(len(image_files) * 0.2)
val_set = random.sample(image_files, val_set_account)
train_set = list(set(image_files) - set(val_set))

test_account = int(len(image_files) * 0.1)
test_set = random.sample(image_files, test_account)

with open(path_val,"w") as f:
    for index in val_set:
        filename = index.split(".")[0]
        f.write(filename+'\n')

with open(path_train,"w") as f:
    for index in train_set:
        filename = index.split(".")[0]
        f.write(filename+'\n')

with open(path_test,"w") as f:
    for index in test_set:
        filename = index.split(".")[0]
        f.write(filename+'\n')

# with open(file, "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  #去掉列表中每一个元素的换行符
#         print(line)
#         shutil.copy(line,path_T)