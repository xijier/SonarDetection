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
        contourData_0 = np.array([(0, 0), (64, 0), (64, 64), (0, 64)])
        cropped = img[ymin_value:ymax_value, xmin_value:xmax_value]
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = cropped.shape[0:2]
        img_cropped_resized = cv2.resize(cropped, (int(y), int(x)))
        img2 = img_cropped_resized.copy()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        roi = img1[ymin_value:ymax_value, xmin_value:xmax_value]
        # 3. 创建掩膜mask
        # img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 将图片灰度化，如果在读取logo时直接灰度化，该步骤可省略
        # cv2.THRESH_BINARY：如果一个像素值低于200，则像素值转换为255（白色色素值），否则转换成0（黑色色素值）
        # 即有内容的地方为黑色0，无内容的地方为白色255.
        # 白色的地方还是白色，除了白色的地方全变成黑色
        ret, mask = cv2.threshold(img2, 175, 255, cv2.THRESH_BINARY)  # 阙值操作
        mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
        dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合
        combine = cv2.addWeighted(roi, 0.2, dst, 0.8, 1)
        img1[ymin_value:ymax_value, xmin_value:xmax_value] = combine  # 将融合后的区域放进原图
        img_new_add = img1.copy()  # 对处理后的图像进行拷贝
        tempList = []
        tempList.append(obj_name)
        tempList.append(img_cropped_resized)
        croppedList.append(tempList)
    return croppedList,img_new_add


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
    print(torch.cuda.is_available())


    path = "E:\kg\data\SONAR_VOC_Gray\VOC2007\images/train"
    data_xml_path = 'E:\kg\data\SONAR_VOC_Gray\VOC2007\Annotations'

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

    # for root, dirs, files in os.walk(data_xml_file):
    #     for file in files:
    #         # 获取文件路径
    #         data_xml_path = os.path.join(root, file)
    #         DOMTree = xml.dom.minidom.parse(data_xml_path)
    #         data = DOMTree.documentElement
    #         nodelist = data.getElementsByTagName("object")
    #         for object in nodelist:
    #             name = object.getElementsByTagName("name")
    #             obj_name = name[0].childNodes[0].nodeValue
    #             if obj_name not in labellist:
    #                 labellist.append(obj_name)
    # for item in labellist:
    #     print(item)

