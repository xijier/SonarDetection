import os,shutil
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parse
import xml.dom.minidom

def txt_xml(img_path,img_name,txt_path,img_txt,xml_path,img_xml):
    #读取txt的信息
    clas=[]
    img=cv2.imread(os.path.join(img_path,img_name))
    imh, imw = img.shape[0:2]
    txt_img=os.path.join(txt_path,img_txt)
    with open(txt_img,"r") as f:
        #next(f)
        for line in f.readlines():
            line = line.strip('\n')
            list = line.split(" ")
            # print(list)
            clas.append(list)
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = '1'
    node_filename = SubElement(node_root, 'filename')
    #图像名称
    node_filename.text = "E:\kg\ssd.pytorch-master\data\sonar\JPEGImages_1\\"+img_name
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
    #图像文件夹所在位置
    img_path = r"E:\kg\ssd.pytorch-master\data\sonar\JPEGImages"
    # #标注文件夹所在位置
    # txt_path=r"E:\kg\ssd.pytorch-master\data\sonar\label_lee"
    # #txt转化成xml格式后存放的文件夹
    # xml_path=r"E:\kg\ssd.pytorch-master\data\sonar\Annotations"
    # for img_name in os.listdir(img_path):
    #     print(img_name)
    #     img_xml=img_name.split(".")[0]+".xml"
    #     img_txt=img_name.split(".")[0]+".txt"
    #     txt_xml(img_path, img_name, txt_path, img_txt,xml_path, img_xml)


    # fiename = "E:\kg\data\SONAR_VOC\VOC2007\Annotations/000005.xml"
    # DOMTree = xml.dom.minidom.parse(fiename)
    # data = DOMTree.documentElement
    # nodelist = data.getElementsByTagName("filename")
    # for item in nodelist:
    #     item.firstChild.data ="000005.jpg"
    # with open(os.path.join(fiename), 'w') as fh:
    #     DOMTree.writexml(fh)

    # data_xml_file = "E:\kg\data\SONAR_VOC\VOC2007/Annotations/"
    # for root, dirs, files in os.walk(data_xml_file):
    #     for file in files:
    #         # 获取文件路径
    #         data_xml_path = os.path.join(root, file)
    #         file.split(".")[0]+".jpg"
    #         DOMTree = xml.dom.minidom.parse(data_xml_path)
    #         data = DOMTree.documentElement
    #         nodelist = data.getElementsByTagName("filename")
    #         for item in nodelist:
    #             print(item.firstChild.data)
    #             item.firstChild.data = file.split(".")[0]+".jpg"
    #             print(item.firstChild.data)
    #         with open(os.path.join(data_xml_path), 'w') as fh:
    #             DOMTree.writexml(fh)




