import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import random
from lxml.etree import Element, SubElement, tostring
import turtle
import torch
# "aeroplane": 1, ->arrow
# "bicycle": 2, ->cycle
# "boat": 4, -> heart
# "bottle": 5, ->quadstar
# "chair": 9, -> triangle
# "motorbike": 14, -> start
# "person": 15 ->square

def mergeImg(inputImg, maskImg, contourData, drawPosition):
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
    input_roi = outPutImg[drawPosition[1]:drawPosition[1] + maskImg_h, drawPosition[0]:drawPosition[0] + maskImg_w]
    imgMask_array = np.zeros((maskImg_h, maskImg_w, maskImg.shape[2]), dtype=np.uint8)
    # triangles_list = [np.zeros((len(contourData), 2), int)]
    triangles_list = [contourData]
    cv2.fillPoly(imgMask_array, triangles_list, color=(1, 1, 1))
    cv2.fillPoly(input_roi, triangles_list, color=(0, 0, 0))
    # cv2.imshow('imgMask_array', imgMask_array)
    imgMask_array = imgMask_array * maskImg
    output_ori = input_roi + imgMask_array
    outPutImg[drawPosition[1]:drawPosition[1] + maskImg_h, drawPosition[0]:drawPosition[0] + maskImg_w] = output_ori
    triangles_list[0][:, 0] = contourData[:, 0] + drawPosition[0]
    triangles_list[0][:, 1] = contourData[:, 1] + drawPosition[1]
    outContourData = triangles_list[0]

    return outPutImg, outContourData  # ,outRectData

def getCrops(nodelist, img):
    croppedList = []
    for object in nodelist:
        name = object.getElementsByTagName("name")
        #print(name[0].childNodes[0].nodeValue)
        obj_name= name[0].childNodes[0].nodeValue
        bndbox = object.getElementsByTagName("bndbox")
        xmin = bndbox[0].getElementsByTagName("xmin")
        xmin_value = int(xmin[0].childNodes[0].nodeValue)
        ymin = bndbox[0].getElementsByTagName("ymin")
        ymin_value = int(ymin[0].childNodes[0].nodeValue)
        xmax = bndbox[0].getElementsByTagName("xmax")
        xmax_value = int(xmax[0].childNodes[0].nodeValue)
        ymax = bndbox[0].getElementsByTagName("ymax")
        ymax_value = int(ymax[0].childNodes[0].nodeValue)
        #print(xmin_value, ymin_value, xmax_value, ymax_value)
        #cv2.rectangle(img, (xmin_value, ymin_value), (xmax_value, ymax_value), (255, 255, 255), 1)  # x0,y0 x1,y1

        # "aeroplane": 1, ->arrow
        # "bicycle": 2, ->cycle
        # "boat": 4, -> heart
        # "bottle": 5, ->quadstar
        # "chair": 9, -> triangle
        # "motorbike": 14, -> star
        # "person": 15 ->square
        contourData_0 = np.array([(0, 0), (64, 0), (64, 64), (0, 64)])
        if obj_name == "arrow":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "arrow.jpg")
        if obj_name == "cycle":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "cycle.jpg")
        if obj_name == "heart":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "heart.jpg")
        if obj_name == "quadstar":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "quadstar.jpg")
        if obj_name == "triangle":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "triangle.jpg")
        if obj_name == "star":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "star.jpg")
        if obj_name == "square":
            img_obj = cv2.imread("E:\kg\data\SONAR_VOC_2\VOC2007/target/target_1/" + "square.jpg")

        img,outContourData_0 = mergeImg(img, img_obj, contourData_0,(xmin_value, ymin_value))
        cropped = img[ymin_value:ymax_value, xmin_value:xmax_value]
        x, y = cropped.shape[0:2]
        img_cropped_resized = cv2.resize(cropped, (int(y), int(x)))
        tempList = []
        tempList.append(obj_name)
        tempList.append(img_cropped_resized)
        croppedList.append(tempList)

    return croppedList,img


def verifyCrops(imageNameList):
    for imageName in imageNameList:
        imagepath = "E:\kg\data\SONAR_VOC_2\VOC2007\JPEGImages/" + imageName
        img_obj = cv2.imread(imagepath)
        data_xml_path = "E:/kg/data/SONAR_VOC_2/VOC2007/Annotations/" + imageName.split(".")[0] + ".xml"

        DOMTree = xml.dom.minidom.parse(data_xml_path)
        data = DOMTree.documentElement
        nodelist = data.getElementsByTagName("object")
        croppedList, img_obj = getCrops(nodelist, img_obj)
        cv2.imwrite(imagepath, img_obj)
        print(imageName)
        #cv2.imshow("Image", img_obj)
        #cv2.waitKey(0)

if __name__ == '__main__':

    # imgpath = "E:\kg\data\SONAR_VOC_2\VOC2007\JPEGImages/000041.jpg"
    # data_xml_path = 'E:\kg\data\SONAR_VOC_2\VOC2007\Annotations/000041.xml'
    # img = cv2.imread(imgpath)
    # DOMTree = xml.dom.minidom.parse(data_xml_path)
    # data = DOMTree.documentElement
    # nodelist = data.getElementsByTagName("object")
    # croppedList,img_obj = getCrops(nodelist,img)
    # cv2.imwrite(imgpath,img_obj)

    path = "E:\kg\data\SONAR_VOC_2\VOC2007\images/val"
    data_xml_path = 'E:\kg\data\SONAR_VOC_2\VOC2007\Annotations'

    imageNameList = os.listdir(path)
    for imgName in imageNameList:
        imgpath = os.path.join(path, imgName)
        img = cv2.imread(imgpath)
        data_xml_name = imgName.split(".")[0]+".xml"
        data_xml_path_name = os.path.join(data_xml_path, data_xml_name)
        DOMTree = xml.dom.minidom.parse(data_xml_path_name)
        data = DOMTree.documentElement
        nodelist = data.getElementsByTagName("object")
        croppedList, img_obj = getCrops(nodelist, img)
        cv2.imwrite(imgpath, img_obj)

    # verifyCrops(imageNameList)
    # print("complete.")
    # data_xml_file = 'E:\kg\data\SONAR_VOC_2\VOC2007\Annotations'
    #
    # for root, dirs, files in os.walk(data_xml_file):
    #     for file in files:
    #         # 获取文件路径
    #         data_xml_path = os.path.join(root, file)
    #         DOMTree = xml.dom.minidom.parse(data_xml_path)
    #         data = DOMTree.documentElement
    #         nodelist = data.getElementsByTagName("object")

