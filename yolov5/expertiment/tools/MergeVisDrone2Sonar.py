import os
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
import random
from lxml.etree import Element, SubElement, tostring

def blurLabel(img2):
    img2 = cv2.resize(img2, (64, 64))
    roi = img2[101:165, 204:268]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 将图片灰度化，如果在读取logo时直接灰度化，该步骤可省略
    ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)  # 阙值操作
    mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合
    combine = cv2.addWeighted(roi, 0.6, dst, 0.4, 1)

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
        if (x > 5) & (y > 5):
            img_cropped_resized = cv2.resize(cropped, (int(y), int(x)))
            tempList = []
            tempList.append(obj_name)
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


    combine = cv2.resize(combine, (maskImg_h, maskImg_w))
    outPutImg[drawPosition[1]:drawPosition[1] + maskImg_h, drawPosition[0]:drawPosition[0] + maskImg_w] = combine

    #cv2.imshow('img_new_add', outPutImg)
    #cv2.waitKey(0)

    triangles_list[0][:, 0] = contourData[:, 0] + drawPosition[0]
    triangles_list[0][:, 1] = contourData[:, 1] + drawPosition[1]
    outContourData = triangles_list[0]

    return outPutImg, outContourData  # ,outRectData


def getrandomPixCenter(img, count, shrink_value = 64):
    w = img.shape[1]
    h = img.shape[0]
    w_list = random.sample(range(shrink_value, w - shrink_value), count)
    h_list = random.sample(range(shrink_value, h - shrink_value), count)
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

    path = os.path.abspath('..')
    VOV_Annotations_Path = "E:\kg\data\VisDrone2020_yolo\VisDrone2020-DET-train/annotations_voc/"
    VOV_JPEGImages_Path = "E:\kg\data\VisDrone2020_yolo\VisDrone2020-DET-train\images/"

    files_data_path = os.listdir(VOV_Annotations_Path)
    files_img_obj = os.listdir(VOV_JPEGImages_Path)

    for imageName in files_img_obj:
        data_path = VOV_Annotations_Path + imageName.split(".")[0]+".xml"
        #print(imageName)
        #print(data_path)
        img_obj = cv2.imread(VOV_JPEGImages_Path + imageName)
        DOMTree = xml.dom.minidom.parse(data_path)
        data = DOMTree.documentElement
        nodelist = data.getElementsByTagName("object")

        files = os.listdir("JPEGImages_pool")
        file = files[random.randint(0, len(files) - 1)]
        img_sonar = cv2.imread("JPEGImages_pool/" + file)

        croppedList = getCrops(nodelist, img_obj)

        if len(croppedList) >10:
            croppedList = random.sample(croppedList, 10)

        shrink_value = 32
        # 获得图片中的随机目标（len(croppedList)）个X,Y 坐标
        centerpix = getrandomPixCenter(img_sonar, len(croppedList), shrink_value)
        shrink_size = (shrink_value, shrink_value)
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
            outPutImg, outContourData_0 = mergeImg(outPutImg,img_sonar, shrink_target, contourData_0,
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
        imagePath = "E:\kg\data\SONAR_VisDrone2020/train/JPEGImages/"
        xml_path = "E:\kg\data\SONAR_VisDrone2020/train/Annotations/"
        img_xml = imageName.split(".")[0] + ".xml"
        cv2.imwrite(imagePath + imageName, outPutImg)
        to_xml(imagePath, imageName, xml_path, img_xml, clas)
        print(imageName)
    print("complete")