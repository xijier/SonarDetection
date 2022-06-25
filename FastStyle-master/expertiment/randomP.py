# -*- coding:utf-8 -*-
#yys
import random


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

path="E:\kg\FastStyle-master\images\content/1-airplane_train_30.jpg"
image=cv2.imread(path,1)

#for i in range(5):
#    image=Drawing_Random_Lines(image)
# for i in range(5):
#     image=Drawing_Random_circle(image)

for j in range(10):
    image = cv2.imread(path, 1)
    for i in range(12):
        image=Drawing_Random_rect(image)
    cv2.imshow("image"+ str(j), image)
#image=Drawing_Random_rect(image)
#image=Drawing_Random_shap(image)
#cv2.imshow("image",image)
cv2.waitKey(0)
#cv2.imwrite("E:\kg\FastStyle-master\images\content/1-airplane_train_30_3.jpg",image)
