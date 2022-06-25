# -*- coding:utf-8 -*-
#yys
import random
import os

#使用cv2以及PIL库随机生成汉字以及线条
import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw



def Gaussnoise_func(image, mean=0, var=0.005):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image/255, dtype=float)                    #将像素值归一
    noise = np.random.normal(mean, var ** 0.5, image.shape)     #产生高斯噪声
    out = image + noise                                         #直接将归一化的图片与噪声相加

    '''
    将值限制在(-1/0,1)间，然后乘255恢复
    '''
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def sp_noise(image,prob):
    '''
    手动添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

images_path="E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target/aeroplane/"
save_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation/SaltAndPepper/aeroplane/"
files_img = os.listdir(images_path)
p1 = 0.4
for image_file in files_img:
    image = cv2.imread(images_path+image_file, 1)
    if random.random() < p1:
        image_1 = image.copy()
        img_r = sp_noise(image_1, prob=0.2)
        cv2.imwrite(save_path+"sp_"+image_file, img_r)

#cv2.imwrite("E:\kg\FastStyle-master\images\content/1-airplane_train_30_3.jpg",image)
