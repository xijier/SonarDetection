## 载入所需库
import cv2
import time
import numpy as np
import torch
import os
import random
import shutil

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

labels_path = "E:\kg\data/NWPU VHR-10 dataset\yoloformat/"
images_path ="E:\kg\data/NWPU VHR-10 dataset\positive image set/"
img_save_path  ="E:\kg\data/NWPU VHR-10 dataset/filter/"
target  ="E:\kg\data/NWPU VHR-10 dataset/labels/"
names = ['plane', 'boat', 'tank', 'baseball court', 'tennis court', 'basketball court', 'race court', 'port', 'bridge', 'car']

#names = ['plane', 'boat', 'car']

files_img = os.listdir(images_path)
files_labels = os.listdir(labels_path)
index = 0
uniquelabels= []
for image_name in files_img:
    index = index+1
    label_name = labels_path +image_name.split(".")[0] + ".txt"
    img1 = cv2.imread(images_path+image_name)
    with open(label_name, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_ = line.strip('\n')
            labels_ = line_.split(" ")
            #uniquelabels.append(labels_[0])
            objname = names[int(labels_[0])]
            if objname is not "plane" and objname is not "boat" and objname is not "car":
                print("1")
            else:
                label_rc = target +image_name.split(".")[0] + ".txt"
                with open(label_rc, "a") as f:
                    if labels_[0] == '9':
                        labels_[0] = '2'
                        line = ' '.join(labels_)
                        line = line + '\n'
                    f.write(line)
                    labels = []
                    for item in labels_:
                        labels.append(float(item))
                    labels = np.array([labels])
                    templabels = xywhn2xyxy(labels[:, 1:],img1.shape[1],img1.shape[0])
                    xmin_value = int(templabels[:, 0])
                    ymin_value = int(templabels[:, 1])
                    xmax_value = int(templabels[:, 2])
                    ymax_value = int(templabels[:, 3])
                    cv2.imwrite(img_save_path + image_name, img1)


            # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度

            #img = cv2.putText(img1, objname, (xmax_value, ymin_value), font, 0.5, (255, 255, 0), 1)
            #cv2.rectangle(img1, (xmin_value, ymin_value), (xmax_value, ymax_value), (0, 255, 0), 2)  # x0,y0 x1,y1

            # if objname == "plane":
            #     cv2.imwrite(img_save_path + image_name, img1)
            #     source = label_name
            #     a = image_name.split(".")[0] + ".txt"
            #     target = "E:\kg\data/NWPU VHR-10 dataset/labels/"
            #     target = os.path.join(target,  a)
            #     shutil.copy(source, target)
            # if objname == "boat":
            #     cv2.imwrite(img_save_path + image_name, img1)
            #     source = label_name
            #     a = image_name.split(".")[0] + ".txt"
            #     target = "E:\kg\data/NWPU VHR-10 dataset/labels/"
            #     target = os.path.join(target,  a)
            #     shutil.copy(source, target)
            # if objname == "car":
            #     cv2.imwrite(img_save_path + image_name, img1)
            #     source = label_name
            #     a = image_name.split(".")[0] + ".txt"
            #     target = "E:\kg\data/NWPU VHR-10 dataset/labels/"
            #     target = os.path.join(target,  a)
            #     shutil.copy(source, target)

        #cv2.imshow("3", img1)
        #cv2.waitKey(0)
        #cv2.imwrite(img_save_path + image_name, img1)

#l2 = list(set(uniquelabels))
#print(l2)