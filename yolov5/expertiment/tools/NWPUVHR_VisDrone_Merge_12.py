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

labels_path = "E:\kg\data\SONAR_VisDrone2020_NWPUVHR10\VOC2007\labels/val/"
images_path ="E:\kg\data\SONAR_VisDrone2020_NWPUVHR10\VOC2007\images/val/"
img_save_path  ="E:\kg\data/NWPU VHR-10 dataset/filter/"
target  ="E:\kg\data\SONAR_VisDrone2020_NWPUVHR10\VOC2007\labels/val1/"

#plane 10 -> 0
#bicycle 2 ->1
#bus     8-> 5
#car     3-> 6
#people  1 -> 14
#motor   9 -> 13
#motor   11-> 3
#pedestrian 0-> 14
#'truck   5->6
#'tricycle 6 ->20',
# 'awning-tricycle  7 - >20'
#'van 4' ->21

#names: ['aeroplane 0', 'bicycle 1', 'bird 2', 'boat 3', 'bottle 4', 'bus  5', 'car  6', 'cat  7', 'chair  8', 'cow  9',
#        'diningtable  10', 'dog  11', 'horse  12', 'motorbike  13', 'person  14', 'pottedplant  15', 'sheep  16', 'sofa  17',
#        'train  18', 'tvmonitor  19' ,tricycle 20, van 21]

files_img = os.listdir(images_path)
files_labels = os.listdir(labels_path)

uniquelabels= []
for image_name in files_img:

    label_name = labels_path +image_name.split(".")[0] + ".txt"
    img1 = cv2.imread(images_path+image_name)
    index = 0
    with open(label_name, "r") as file:
        lines = file.readlines()
        for line in lines:

            line_ = line.strip('\n')
            labels_ = line_.split(" ")
            #uniquelabels.append(labels_[0])
            #objname = names[int(labels_[0])]
            label_rc = target +image_name.split(".")[0] + ".txt"
            if index >= (len(lines)/2) :
                with open(label_rc, "a") as f:
                    f.write(line)
                    labels = []
                    for item in labels_:
                        labels.append(float(item))
                    labels = np.array([labels])
            index = index + 1

                    #templabels = xywhn2xyxy(labels[:, 1:],img1.shape[1],img1.shape[0])
                    # xmin_value = int(templabels[:, 0])
                    # ymin_value = int(templabels[:, 1])
                    # xmax_value = int(templabels[:, 2])
                    # ymax_value = int(templabels[:, 3])
                    #cv2.imwrite(img_save_path + image_name, img1)

#l2 = list(set(uniquelabels))
#print(l2)