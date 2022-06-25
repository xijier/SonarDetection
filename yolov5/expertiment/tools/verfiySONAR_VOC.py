import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import random
from lxml.etree import Element, SubElement, tostring

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
        cv2.rectangle(img, (xmin_value, ymin_value), (xmax_value, ymax_value), (255, 255, 0), 1)  # x0,y0 x1,y1
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
        img = cv2.putText(img, obj_name, (xmax_value, ymin_value), font, 0.5, (255, 255, 0), 1)

        cropped = img[ymin_value:ymax_value, xmin_value:xmax_value]
        x, y = cropped.shape[0:2]
        img_cropped_resized = cv2.resize(cropped, (int(y), int(x)))
        tempList = []
        tempList.append(obj_name)
        tempList.append(img_cropped_resized)
        croppedList.append(tempList)
    return croppedList

imageNameList = []
imageNameList.append("000007.jpg")
imageNameList.append("000044.jpg")
imageNameList.append("000046.jpg")
imageNameList.append("000047.jpg")
imageNameList.append("000048.jpg")
imageNameList.append("000050.jpg")
imageNameList.append("000060.jpg")
imageNameList.append("000061.jpg")

img_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\VOC2007\JPEGImages/"
xml_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\VOC2007\Annotations/"

files_img_obj = os.listdir(img_path)

for imageName in files_img_obj:

    img_obj = cv2.imread(img_path + imageName)
    data_xml_path = xml_path + imageName.split(".")[0] + ".xml"
    DOMTree = xml.dom.minidom.parse(data_xml_path)
    data = DOMTree.documentElement
    nodelist = data.getElementsByTagName("object")
    croppedList = getCrops(nodelist, img_obj)
    cv2.imshow("Image",img_obj)
    cv2.waitKey(0)
