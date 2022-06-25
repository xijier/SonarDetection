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
        # cv2.rectangle(img, (xmin_value, ymin_value), (xmax_value, ymax_value), colors, 1)  # x0,y0 x1,y1
        cropped = img[ymin_value:ymax_value, xmin_value:xmax_value]
        x, y = cropped.shape[0:2]
        img_cropped_resized = cv2.resize(cropped, (int(y), int(x)))
        tempList = []
        tempList.append(obj_name)
        tempList.append(img_cropped_resized)
        croppedList.append(tempList)
    return croppedList


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


def getrandomPixCenter(img, count,shrink_size = (32,32)):
    w = img.shape[1]
    h = img.shape[0]
    w_list = random.sample(range(shrink_size[0], w - shrink_size[0]), count)
    h_list = random.sample(range(shrink_size[1], h - shrink_size[1]), count)
    centerpiece = np.array(w_list)
    centerpiece = np.row_stack((centerpiece, h_list))
    centerpiece = centerpiece.T
    return centerpiece

def to_xml(img_path,img_name,xml_path,img_xml,clas):
    #读取txt的信息
    img=cv2.imread(os.path.join(img_path,img_name))
    imh, imw = img.shape[0:2]
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = '1'
    node_filename = SubElement(node_root, 'filename')
    #图像名称
    node_filename.text = img_path + img_name
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(imw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(imh)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(len(clas)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(clas[i][0])
        node_pose=SubElement(node_object, 'pose')
        node_pose.text="Unspecified"
        node_truncated=SubElement(node_object, 'truncated')
        node_truncated.text="truncated"
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(clas[i][1])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(clas[i][2])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(clas[i][3]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(clas[i][4]))
    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    img_newxml = os.path.join(xml_path, img_xml)
    file_object = open(img_newxml, 'wb')
    file_object.write(xml)
    file_object.close()

if __name__ == "__main__":

    path = os.path.abspath('.')

    img_folder_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1\content_styled/"
    target_img_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/traget/JPEGImages/"
    target_xml_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/traget/Annotations/"
    JPEGImages_pool = "E:\kg\data\SONARIMG_pool/"
    lables = ['aeroplane','bicycle','car','person','ship']
    #4种缩放类别
    shrink_size_list = [(64, 64),(32, 64),(32, 32),(64, 32)]
    #产生11000张带有目标的声呐图片
    for index_img in range(0,11000):
        #每次取2个目标类别
        target_labels = random.sample(lables, k=2)
        croppedList = []
        for label in target_labels:
            image_path = img_folder_path+ label
            image_list = os.listdir(image_path)
            #每次取2个目标
            image_label_list = random.sample(image_list, k=2)
            for img_name in image_label_list:
                img = cv2.imread(image_path+"/"+img_name)
                shrink_size = random.sample(shrink_size_list, k=1)[0]
                img_cropped_resized = cv2.resize(img, (shrink_size[0],shrink_size[1]))
                tempList = []
                tempList.append(label)
                tempList.append(img_cropped_resized)
                croppedList.append(tempList)
        files = os.listdir(JPEGImages_pool)
        file = files[random.randint(0, len(files) - 1)]
        img_sonar = cv2.imread(JPEGImages_pool+file)
        centerpix = getrandomPixCenter(img_sonar, len(croppedList), (64, 64))
        outPutImg = img_sonar
        clas = []
        for index, item in enumerate(croppedList):
            h_0 = item[1].shape[0]
            w_0 = item[1].shape[1]
            contourData_0 = np.array([(0, 0), (w_0, 0), (w_0, h_0), (0, h_0)])
            outPutImg, outContourData_0 = mergeImg(outPutImg, item[1], contourData_0,
                                                    (centerpix[index][0], centerpix[index][1]))
            tempList = []
            tempList.append(croppedList[index][0])
            tempList.append(contourData_0[0][0])  # xmin
            tempList.append(contourData_0[0][1])  # ymin
            tempList.append(contourData_0[2][0])  # xmax
            tempList.append(contourData_0[2][1])  # ymax
            clas.append(tempList)

        imageName = str(index_img) + ".jpg"
        img_xml = target_xml_path + str(index_img) + ".xml"
        cv2.imwrite(target_img_path + imageName, outPutImg)
        to_xml(target_img_path, imageName, target_xml_path, img_xml, clas)
        print(imageName)
    print("complete")