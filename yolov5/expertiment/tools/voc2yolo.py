import xml.etree.ElementTree as ET
import os
import random
from shutil import copyfile

# box [xmin,ymin,xmax,ymax]
def convert(size, box):
    x_center = (box[2] + box[0]) / 2.0
    y_center = (box[3] + box[1]) / 2.0
    # 归一化
    x = x_center / size[0]
    y = y_center / size[1]
    # 求宽高并归一化
    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]
    return (x, y, w, h)


def convert_annotation(xml_paths, yolo_paths, classes):
    xml_files = os.listdir(xml_paths)
    # 生成无序文件列表
    print(f'xml_files:{xml_files}')
    for file in xml_files:
        xml_file_path = os.path.join(xml_paths, file)
        yolo_txt_path = os.path.join(yolo_paths, file.split(".")[0]
                                     + ".txt")
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find("size")
        # 获取xml的width和height的值
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        # object标签可能会存在多个，所以要迭代
        with open(yolo_txt_path, 'w') as f:
            for obj in root.iter("object"):
                difficult = obj.find("difficult").text
                # 种类类别
                cls = obj.find("name").text
                if cls not in classes or difficult == 1:
                    continue
                # 转换成训练模式读取的标签
                cls_id = classes.index(cls)
                #cls_id = 6
                xml_box = obj.find("bndbox")
                box = (float(xml_box.find("xmin").text), float(xml_box.find("ymin").text),
                       float(xml_box.find("xmax").text), float(xml_box.find("ymax").text))
                boxex = convert((w, h), box)
                # yolo标准格式类别 x_center,y_center,width,height
                f.write(str(cls_id) + " " + " ".join([str(s) for s in boxex]) + '\n')

def split2trian_val(base_folder,yolo_txt_dir):
    #base_folder = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/target_origin_noVOC/"
    img_path = base_folder + "JPEGImages/"
    image_files = os.listdir(img_path)
    val_set_account = int(len(image_files) * 0.2)
    train_set_account = int(len(image_files) * 0.8)
    val_set = random.sample(image_files, val_set_account)
    train_set = list(set(image_files) - set(val_set))
    yolo_img_val = base_folder + "images/val/"
    yolo_labels_val = base_folder + "labels/val/"
    yolo_img_train = base_folder + "images/train/"
    yolo_labels_train = base_folder + "labels/train/"
    for image_file in val_set:
        #print(image_file)
        text_file = image_file.split(".")[0] +".txt"
        copyfile(img_path+image_file, yolo_img_val+image_file)
        copyfile(yolo_txt_dir + text_file, yolo_labels_val + text_file)

    for image_file in train_set:
        #print(image_file)
        text_file = image_file.split(".")[0] +".txt"
        copyfile(img_path+image_file, yolo_img_train+image_file)
        copyfile(yolo_txt_dir + text_file, yolo_labels_train + text_file)

if __name__ == "__main__":
    # 数据的类别
    classes_train = ['aeroplane','bicycle','car','person','ship']
    #classes_train = ['ship']
    # xml存储地址
    xml_dir = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/target_origin_noVOC/Annotations/"
    # yolo存储地址
    yolo_txt_dir = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/target_origin_noVOC/yolo/"
    # voc转yolo
    base_folder = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer\VOC2007_Style_1/target_origin_noVOC/"
    convert_annotation(xml_paths=xml_dir, yolo_paths=yolo_txt_dir,
                        classes=classes_train)

    split2trian_val(base_folder,yolo_txt_dir)
