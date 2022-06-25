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
    w1 = int(w / 10)
    radius = random.randint(5, w1+5)
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

names = ['aeroplane','bicycle','car','person','ship']

images_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation/ship/origin/"

save_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation/ship/"

p1 = 0.4
p2 = 0.4
p3 = 0.4

N = 8
random.random()

aug_name = ['line','circle','rectangle']

files_img = os.listdir(images_path)

for image_file in files_img:
    image = cv2.imread(images_path+image_file, 1)
    if random.random() < p1:
        image_1 = image.copy()
        for i in range(N):
            image_1 = Drawing_Random_Lines(image_1)
        #cv2.imshow("image1", image_1)
        #cv2.waitKey(0)
        save_file = save_path + '/line/'+ str(N)+"/"
        #print(save_file+"1_"+image_file)
        cv2.imwrite(save_file+"1_"+image_file, image_1)
    if random.random() < p2:
        image_2 = image.copy()
        for i in range(N):
            image_2 = Drawing_Random_circle(image_2)
        #cv2.imshow("image2", image_2)
        save_file = save_path + '/circle/' + str(N) + "/"
        cv2.imwrite(save_file + "2_" + image_file, image_2)
    if random.random() < p3:
        image_3 = image.copy()
        for i in range(N):
            image_3 = Drawing_Random_rect(image_3)
        #cv2.imshow("image3", image_3)
        save_file = save_path + '/rectangle/' + str(N) + "/"
        cv2.imwrite(save_file + "3_" + image_file, image_3)

    #cv2.waitKey(0)

#image=cv2.imread(images_path,1)

#for i in range(5):
#    image=Drawing_Random_Lines(image)
# for i in range(5):
#     image=Drawing_Random_circle(image)

# for i in range(3):
#     image=Drawing_Random_rect(image)
#image=Drawing_Random_rect(image)
#image=Drawing_Random_shap(image)
#cv2.imshow("image",image)
#cv2.waitKey(0)
#cv2.imwrite("E:\kg\FastStyle-master\images\content/1-airplane_train_30_3.jpg",image)
