# -*- coding:utf-8 -*-
#yys
import random
import os

#使用cv2以及PIL库随机生成汉字以及线条
import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

# 随机生成线段
def Drawing_Random_Lines(image):

    h, w = image.shape[0], image.shape[1]
    (x1,y1 ) = ( random.randint(0, w) ,random.randint(0,h) )
    (x2, y2) = ( random.randint(0, w), random.randint(0, h))
    RGB=(random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
    cv2.line(image,(x1,y1 ), (x2, y2),RGB, random.randint(1, 10), 8)
    # 将BGR转换为RGB
    image=image[...,::-1]

    image = Image.fromarray(image)
    #image.show()
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image

def Drawing_Random_circle(image):

    h, w = image.shape[0], image.shape[1]
    (x1,y1 ) = ( random.randint(0, w) ,random.randint(0,h) )
    (x2, y2) = ( random.randint(0, w), random.randint(0, h))
    RGB=(random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
    #cv2.rectangle(image,(20,20),(50,50),random.randint(0, 255),-1)
    radius = random.randint(5, 15)
    cv2.circle(image,(x1,y1),radius,RGB,-1)

    #cv2.line(image,(x1,y1 ), (x2, y2),RGB, random.randint(1, 10), 8)
    # 将BGR转换为RGB
    image=image[...,::-1]
    image = Image.fromarray(image)
    #image.show()
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image

def Drawing_Random_rect(image):

    h, w = image.shape[0], image.shape[1]
    w1 = int(w/5)
    h1 = int(h/5)
    (x1,y1 ) = ( random.randint(0, w) ,random.randint(0, h) )
    (x2, y2) = ( random.randint(0, w1), random.randint(0, h1))
    RGB=(random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
    cv2.rectangle(image,(x1,y1),(x1+w1,y1+h1),RGB,-1)
    radius = random.randint(5, 15)
    #cv2.circle(image,(x1,y1),radius,RGB,-1)
    #cv2.line(image,(x1,y1 ), (x2, y2),RGB, random.randint(1, 10), 8)
    # 将BGR转换为RGB
    image=image[...,::-1]
    image = Image.fromarray(image)
    #image.show()
    image=np.array(image)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image

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

images_path="E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target/ship/"
save_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation/gauss/ship/"
files_img = os.listdir(images_path)
p1 = 0.4
for image_file in files_img:
    image = cv2.imread(images_path+image_file, 1)
    if random.random() < p1:
        image_1 = image.copy()
        img_r = Gaussnoise_func(image_1, 1 / 10, 15 / 100)
        cv2.imwrite(save_path+"gauss_"+image_file, img_r)

#cv2.imwrite("E:\kg\FastStyle-master\images\content/1-airplane_train_30_3.jpg",image)
