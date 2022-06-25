import os
from xml.dom.minidom import parse
import xml.dom.minidom

classes = ["aeroplane", "bicycle", "boat", "bottle","chair","motorbike", "person"]


classesmap = {'aeroplane': 'arrow', 'bicycle': 'cycle', 'boat': 'heart',
           'bottle': 'quadstar', 'chair': 'triangle', 'motorbike': 'star',
           'person': 'square'}


data_xml_path = "E:\kg\data\SONAR_VOC_2\VOC2007\Annotations"

# data_xml_file = data_xml_path+"/000009.xml"
# DOMTree = xml.dom.minidom.parse(data_xml_file)
# data = DOMTree.documentElement
# nodelist = data.getElementsByTagName("object")
# for object in nodelist:
#     name = object.getElementsByTagName("name")
#     obj_name = name[0].childNodes[0].nodeValue
#     name[0].childNodes[0].nodeValue = classesmap[obj_name]
    # with open(os.path.join(data_xml_file), 'w') as fh:
    #      DOMTree.writexml(fh)
labellist = []
for root, dirs, files in os.walk(data_xml_path):
    for file in files:
        # 获取文件路径
        data_xml_file = os.path.join(data_xml_path, file)
        DOMTree = xml.dom.minidom.parse(data_xml_file)
        data = DOMTree.documentElement
        nodelist = data.getElementsByTagName("object")
        for object in nodelist:
            name = object.getElementsByTagName("name")
            obj_name = name[0].childNodes[0].nodeValue
            #name[0].childNodes[0].nodeValue = classesmap[obj_name]
            #with open(os.path.join(data_xml_file), 'w') as fh:
            #     DOMTree.writexml(fh)
            if obj_name not in labellist:
                labellist.append(obj_name)

for item in labellist:
     print(item)