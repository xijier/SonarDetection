import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import random
from lxml.etree import Element, SubElement, tostring
import torch

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def getCrops(label_name, img):

    croppedList = []
    with open(label_name, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_ = line.strip('\n')
            labels_ = line_.split(" ")
            labels = []
            for item in labels_:
                labels.append(float(item))
            labels = np.array([labels])
            templabels = xywhn2xyxy(labels[:, 1:],img.shape[1],img.shape[0])
            xmin_value = int(templabels[:, 0])
            ymin_value = int(templabels[:, 1])
            xmax_value = int(templabels[:, 2])
            ymax_value = int(templabels[:, 3])
            cv2.rectangle(img, (xmin_value, ymin_value), (xmax_value, ymax_value), (255,0,0), 1)  # x0,y0 x1,y1
            cropped = img[ymin_value:ymax_value, xmin_value:xmax_value]
            x, y = cropped.shape[0:2]
            img_cropped_resized = cv2.resize(cropped, (int(y), int(x)))
            tempList = []
            tempList.append(labels_[0])
            tempList.append(img_cropped_resized)
            croppedList.append(tempList)
    return croppedList


def mergeImg(inputImg, orgimg ,maskImg, contourData, drawPosition):
    '''
    :param inputImg: 输入的图像
    :param maskImg: 输入的模板图像
    :param contourData: 输入的模板中轮廓数据 numpy 形式如[(x1,y1),(x2,y2),...,]
    :param drawPosition: （x,y） 大图中要绘制模板的位置,以maskImg左上角为起始点
    :return: outPutImg：输出融合后的图像
             outContourData: 输出轮廓在inputImg的坐标数据
             outRectData: 输出轮廓的矩形框在inputImg的坐标数据
    '''
    # 通道需要相等
    if (inputImg.shape[2] != maskImg.shape[2]):
        print("inputImg shape != maskImg shape")
        return
    inputImg_h = inputImg.shape[0]
    inputImg_w = inputImg.shape[1]
    maskImg_h = maskImg.shape[0]
    maskImg_w = maskImg.shape[1]
    # inputImg图像尺寸不能小于maskImg
    if (inputImg_h < maskImg_h or inputImg_w < maskImg_w):
        print("inputImg size < maskImg size")
        return
    # 画图的位置不能超过原始图像
    if (((drawPosition[0] + maskImg_w) > inputImg_w) or ((drawPosition[1] + maskImg_h) > inputImg_h)):
        print("drawPosition + maskImg > inputImg range")
        return

    outPutImg = inputImg.copy()
    triangles_list = [contourData]
    gray = cv2.cvtColor(maskImg, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(gray, cv2.cv2.COLOR_GRAY2BGR)
    if random.random() < 0.5:
        combine = cv2.add(cv2.resize(orgimg, (200, 200)), cv2.resize(img2, (200, 200)))
    else:
        combine = cv2.addWeighted(cv2.resize(orgimg, (200, 200)), 0.4, cv2.resize(img2, (200, 200)), 0.6, 0)
    combine = cv2.resize(combine, (maskImg_w, maskImg_h))
    outPutImg[drawPosition[1]:drawPosition[1] + maskImg_h, drawPosition[0]:drawPosition[0] + maskImg_w] = combine

    triangles_list[0][:, 0] = contourData[:, 0] + drawPosition[0]
    triangles_list[0][:, 1] = contourData[:, 1] + drawPosition[1]
    outContourData = triangles_list[0]

    return outPutImg, outContourData  # ,outRectData

def to_labelstxt(img, clas,save_lable_path):
    h = img.shape[0]
    w = img.shape[1]
    save_lable_path
    with open(save_lable_path,"w") as f:

        for item in range(len(clas)):
            xmin =clas[item][1]
            ymin = clas[item][2]
            xmax = clas[item][3]
            ymax = clas[item][4]
            b =(xmin,xmax,ymin,ymax)
            y = convert((w, h),b)
            values = list(y)
            values.insert(0,clas[item][0])
            for index,value in enumerate(values):
                if index== (len(values) -1):
                    f.write(str(value)+'\n')
                else:
                    f.write(str(value) + ' ')

def getrandomPixCenter(img, count,shrink_size = (32,32)):
    w = img.shape[1]
    h = img.shape[0]
    w_list = random.sample(range(shrink_size[0], w - shrink_size[0]), count)
    h_list = random.sample(range(shrink_size[1], h - shrink_size[1]), count)
    centerpiece = np.array(w_list)
    centerpiece = np.row_stack((centerpiece, h_list))
    centerpiece = centerpiece.T
    return centerpiece

if __name__ == "__main__":

    path = os.path.abspath('.')
    SONARIMG_pool = "E:\kg\data\SONARIMG_pool/"
    imgs_path = "E:\kg\data/NWPU VHR-10 dataset/filter/"
    labels_path = "E:\kg\data/NWPU VHR-10 dataset\labels/"
    save_img_path = "E:\kg\data/NWPU VHR-10 dataset/filter1/"
    save_labels_path ="E:\kg\data/NWPU VHR-10 dataset\labels1/"
    files_img_obj = os.listdir(imgs_path)
    #names: ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    for imageName in files_img_obj:
        label_name = labels_path + imageName.split(".")[0]+".txt"
        img_obj = cv2.imread(imgs_path + imageName)
        croppedList = getCrops(label_name, img_obj)
        if len(croppedList) >5:
            croppedList = random.sample(croppedList, 6)
        shrink_size = (64, 64)
        files = os.listdir(SONARIMG_pool)
        file = files[random.randint(0, len(files) - 1)]
        img_sonar = cv2.imread(SONARIMG_pool + file)
        # 获得图片中的随机目标（len(croppedList)）个X,Y 坐标
        centerpix = getrandomPixCenter(img_sonar, len(croppedList), shrink_size)
        outPutImg = img_sonar
        # croppedList[index][0] 对应obj_name   croppedList[index][1]对应目标image
        clas = []
        for index, item in enumerate(croppedList):
            # print(item,index)
            shrink_target = cv2.resize(croppedList[index][1], shrink_size, interpolation=cv2.INTER_AREA)
            h_0 = shrink_target.shape[0]
            w_0 = shrink_target.shape[1]
            # contourData = np.array([(57, 7), (107, 30), (107, 120), (62, 122), (2, 95), (9, 32)]) #轮廓数据
            contourData_0 = np.array([(0, 0), (w_0, 0), (w_0, h_0), (0, h_0)])
            outPutImg, outContourData_0 = mergeImg(outPutImg, img_sonar, shrink_target, contourData_0,
                                                   (centerpix[index][0], centerpix[index][1]))
            tempList = []
            tempList.append(croppedList[index][0])
            tempList.append(contourData_0[0][0])  # xmin
            tempList.append(contourData_0[0][1])  # ymin
            tempList.append(contourData_0[2][0])  # xmax
            tempList.append(contourData_0[2][1])  # ymax
            clas.append(tempList)
            #start_point = (contourData_0[0][0], contourData_0[0][1])
            #end_point = (contourData_0[2][0], contourData_0[2][1])
            #cv2.rectangle(outPutImg, start_point, end_point, (255, 255, 0), 2)
        #xml_path = "../SONAR_VOC/Annotations/"
        imageName = imageName.split(".")[0] +"_"+str(shrink_size[0]) +"x"+ str(shrink_size[1]) +".jpg"
        save_lable_path = save_labels_path + imageName.split(".")[0]  +".txt"
        #if random.random() < 0.85:
        to_labelstxt(outPutImg,clas,save_lable_path)
        cv2.imwrite(save_img_path + imageName, outPutImg)

        print(imageName)
    print("complete")
