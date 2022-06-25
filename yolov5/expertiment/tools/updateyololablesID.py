## 载入所需库
import cv2
import time
import numpy as np
import torch
import os
import random

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

#images_path = "E:\kg\data\SONAR_NWPUVHR10\VOC2007\images/train/"
#labels_path = "E:\kg\data\SONAR_NWPUVHR10\VOC2007\labels/train/"

labels_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007\labels/train1/"

names = ['aeroplane','bicycle','boat','car','motorbike','person']
names = ['aeroplane','bicycle','boat','car','motorbike','person','ship']

files_labels = os.listdir(labels_path)
index = 0
for label_name in files_labels:
    label_file = labels_path +label_name
    context =[]
    w_flag = False
    with open(label_file, 'r+') as file:
         lines = file.readlines()
         for line in lines:
             line_ = line.strip('\n')
             labels_ = line_.split(" ")
             #print(labels_[0] + " " + names[int(labels_[0])])
             if labels_[0] == "6":
                 #print(label_name +" " +labels_[0] + " " + names[int(labels_[0])])
                 #file.write("2 "+" ".join([str(s) for s in labels_[1:]])+ '\n')
                 context.append("2 "+" ".join([str(s) for s in labels_[1:]])+ '\n')
                 w_flag = True
    if w_flag == True:
        with open(label_file, 'w+') as file:
            for line in context:
                file.write(line)

    #         for item in labels_:
    #             labels.append(float(item))
    #         labels = np.array([labels])
    #         templabels = xywhn2xyxy(labels[:, 1:],img1.shape[1],img1.shape[0])
    #         xmin_value = int(templabels[:, 0])
    #         ymin_value = int(templabels[:, 1])
    #         xmax_value = int(templabels[:, 2])
    #         ymax_value = int(templabels[:, 3])