import os
from xml.dom.minidom import parse
import xml.dom.minidom
from xml.etree.ElementTree import ElementTree,Element

if __name__ == '__main__':


    data_xml_file = "E:\kg\data\SONAR_VOC_1\VOC2007\Annotations/"
    data_image_file = "E:\kg\data\SONAR_VOC_1\VOC2007\JPEGImages/"
    data_imageSets_file = "E:\kg\data\SONAR_VOC_1\VOC2007\ImageSets\Main/"

    deletTag = "diningtable"
    for root, dirs, files in os.walk(data_xml_file):
        for file in files:
            # 获取文件路径
            data_xml_path = os.path.join(root, file)
            DOMTree = xml.dom.minidom.parse(data_xml_path)
            data = DOMTree.documentElement
            nodelist = data.getElementsByTagName("object")
            for object in nodelist:
                name = object.getElementsByTagName("name")
                obj_name = name[0].childNodes[0].nodeValue
                if obj_name == deletTag:
                    data.removeChild(object)
            with open(os.path.join(data_xml_path), 'w') as fh:
                 DOMTree.writexml(fh)

    imageList = []
    for root, dirs, files in os.walk(data_xml_file):
        for file in files:
            # 获取文件路径
            data_xml_path = os.path.join(root, file)
            DOMTree = xml.dom.minidom.parse(data_xml_path)
            data = DOMTree.documentElement
            nodelist = data.getElementsByTagName("object")
            if len(nodelist) == 0:
                filename = data.getElementsByTagName("filename")
                imageList.append(filename[0].firstChild.data.split(".")[0])
                data_img_path = os.path.join(data_image_file, filename[0].firstChild.data)
                os.remove(data_xml_path)
                os.remove(data_img_path)

    filename = data_imageSets_file + "train.txt"
    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w", encoding="utf-8") as f_w:
        for line in lines:
            line_ = line.strip('\n')
            if imageList.count(line_) > 0:
                continue
            f_w.write(line)

    filename = data_imageSets_file + "val.txt"
    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w", encoding="utf-8") as f_w:
        for line in lines:
            line_ = line.strip('\n')
            if imageList.count(line_) > 0:
                continue
            f_w.write(line)
    print("complete")
    # filename = data_imageSets_file +"horse_train.txt"
    # with open(filename, "r") as file:
    #     for line in file.readlines():
    #         line = line.strip('\n')  # 去掉列表中每一个元素的换行符
    #         splitName = line.split(" ")
    #         if len(splitName)>2:
    #             if splitName[2] == "1":
    #                 print(splitName[0])
