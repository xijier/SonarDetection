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

labels_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/target_origin_noVOC\labels/train/"
images_path ="E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/target_origin_noVOC\images/train/"

#names = ['aeroplane','bicycle','boat','car','motorbike','person']
names = ['aeroplane','bicycle','car','person','ship']
files_img = os.listdir(images_path)
files_labels = os.listdir(labels_path)
index = 0
#cls= []
for image_name in files_img:
    index = index+1
    label_name = labels_path +image_name.split(".")[0] + ".txt"
    img1 = cv2.imread(images_path+image_name)
    with open(label_name, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_ = line.strip('\n')
            labels_ = line_.split(" ")
            labels = []
            for item in labels_:
                labels.append(float(item))
            labels = np.array([labels])
            templabels = xywhn2xyxy(labels[:, 1:],img1.shape[1],img1.shape[0])
            xmin_value = int(templabels[:, 0])
            ymin_value = int(templabels[:, 1])
            xmax_value = int(templabels[:, 2])
            ymax_value = int(templabels[:, 3])
            #cls.append(names[int(labels_[0])])

            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            img = cv2.putText(img1, names[int(labels_[0])], (xmax_value, ymin_value), font, 0.5, (255, 255, 0), 1)
            cv2.rectangle(img1, (xmin_value, ymin_value), (xmax_value, ymax_value), (0, 255, 0), 2)  # x0,y0 x1,y1
            cv2.imshow("1", img1)
            cv2.waitKey(0)

            # if int(labels_[0]) == 0 or int(labels_[0]) ==1:
            #     font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            #     img = cv2.putText(img1, labels_[0], (xmax_value, ymin_value), font, 0.5, (255, 255, 0), 1)
            #     cv2.rectangle(img1, (xmin_value, ymin_value), (xmax_value, ymax_value), (0, 255, 0), 2)  # x0,y0 x1,y1
            #     cv2.imshow("1", img1)
            #     cv2.waitKey(0)

            #cv2.imwrite(img_save_path + image_name, img1)
# new_numbers = list(set(cls))
# print(new_numbers)