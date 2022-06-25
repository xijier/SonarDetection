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

labels_path = "E:\kg\data/NWPU VHR-10 dataset/labels/"
images_path ="E:\kg\data/NWPU VHR-10 dataset/filter/"
save_path = "E:\kg\data/NWPU VHR-10 dataset/targets/"
names = ['aeroplane','bicycle','boat','car','motorbike','person']
files_img = os.listdir(images_path)
files_labels = os.listdir(labels_path)
index = 0
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

            font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
            img = cv2.putText(img1, names[int(labels_[0])], (xmax_value, ymin_value), font, 0.5, (255, 255, 0), 1)

            cv2.rectangle(img1, (xmin_value, ymin_value), (xmax_value, ymax_value), (0, 255, 0), 2)  # x0,y0 x1,y1
            input_roi = img1[ymin_value:ymax_value ,xmin_value:xmax_value]
            #cv2.imshow("1", input_roi)
            path = save_path + str(labels_[0]) + str(index) + '.jpg'
            cv2.imwrite(save_path + str(labels_[0])+ str(index) + '.jpg', input_roi)
            cv2.waitKey(0)