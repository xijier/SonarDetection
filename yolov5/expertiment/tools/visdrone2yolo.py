import os
from os import getcwd
from PIL import Image
import xml.etree.ElementTree as ET
import random

#root_dir = "train/"

#root_dir = "/home/lhw/Gradute/VirDrone/yolov5/VisDrone2019-DET-train/"
root_dir = "E:\kg\data\SONAR_VisDrone2020\VisDrone2020-DET-val\VisDrone2018-DET-val/"

annotations_dir = root_dir+"annotations/"
image_dir = root_dir + "images/"
label_dir = root_dir + "labels/"
# label_dir = root_dir + "images/"    # yolo里面要和图片放到一起
xml_dir = root_dir+"annotations_voc/"  #注意新建文件夹。后续改一下名字，运行完成之后annotations这个文件夹就不需要了。把annotations_命名为annotations
data_split_dir = root_dir + "train_namelist/"

sets = ['train', 'test','val']
class_name = ['ignored regions', 'pedestrian','people','bicycle','car', 'van', 'truck', 'tricycle','awning-tricycle', 'bus','motor','others']

def visdrone2voc(annotations_dir, image_dir, xml_dir):
    for filename in os.listdir(annotations_dir):
        fin = open(annotations_dir + filename, 'r')
        image_name = filename.split('.')[0]
        img = Image.open(image_dir + image_name + ".jpg")
        xml_name = xml_dir + image_name + '.xml'
        with open(xml_name, 'w') as fout:
            fout.write('<annotation>' + '\n')

            fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
            fout.write('\t' + '<filename>' + image_name + '.jpg' + '</filename>' + '\n')

            fout.write('\t' + '<source>' + '\n')
            fout.write('\t\t' + '<database>' + 'VisDrone2018 Database' + '</database>' + '\n')
            fout.write('\t\t' + '<annotation>' + 'VisDrone2018' + '</annotation>' + '\n')
            fout.write('\t\t' + '<image>' + 'flickr' + '</image>' + '\n')
            fout.write('\t\t' + '<flickrid>' + 'Unspecified' + '</flickrid>' + '\n')
            fout.write('\t' + '</source>' + '\n')

            fout.write('\t' + '<owner>' + '\n')
            fout.write('\t\t' + '<flickrid>' + 'Haipeng Zhang' + '</flickrid>' + '\n')
            fout.write('\t\t' + '<name>' + 'Haipeng Zhang' + '</name>' + '\n')
            fout.write('\t' + '</owner>' + '\n')

            fout.write('\t' + '<size>' + '\n')
            fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
            fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
            fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
            fout.write('\t' + '</size>' + '\n')

            fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

            for line in fin.readlines():
                line = line.split(',')
                fout.write('\t' + '<object>' + '\n')
                fout.write('\t\t' + '<name>' + class_name[int(line[5])] + '</name>' + '\n')
                fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
                fout.write('\t\t' + '<truncated>' + line[6] + '</truncated>' + '\n')
                fout.write('\t\t' + '<difficult>' + str(int(line[7])) + '</difficult>' + '\n')
                fout.write('\t\t' + '<bndbox>' + '\n')
                fout.write('\t\t\t' + '<xmin>' + line[0] + '</xmin>' + '\n')
                fout.write('\t\t\t' + '<ymin>' + line[1] + '</ymin>' + '\n')
                # pay attention to this point!(0-based)
                fout.write('\t\t\t' + '<xmax>' + str(int(line[0]) + int(line[2]) - 1) + '</xmax>' + '\n')
                fout.write('\t\t\t' + '<ymax>' + str(int(line[1]) + int(line[3]) - 1) + '</ymax>' + '\n')
                fout.write('\t\t' + '</bndbox>' + '\n')
                fout.write('\t' + '</object>' + '\n')

            fin.close()
            fout.write('</annotation>')

def data_split(xml_dir, data_split_dir):
    trainval_percent = 0.2
    train_percent = 0.9
    total_xml = os.listdir(xml_dir)
    if not os.path.exists(data_split_dir):
        os.makedirs(data_split_dir)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(data_split_dir+'/trainval.txt', 'w')
    ftest = open(data_split_dir+'/test.txt', 'w')
    ftrain = open(data_split_dir+'/train.txt', 'w')
    fval = open(data_split_dir+'/val.txt', 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftest.write(name)
            else:
                fval.write(name)
        else:
            ftrain.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


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

def convert_annotation_voc(xml_dir, label_dir, image_name):
    in_file = open(xml_dir + '%s.xml' % (image_name))
    out_file = open(label_dir + '%s.txt' % (image_name), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_name or int(difficult) == 1:
            continue
        cls_id = class_name.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        if cls_id != 0:  # 忽略掉0类
            if cls_id != 11:  # 忽略掉11类
                out_file.write(str(cls_id - 1) + " " + " ".join([str(a) for a in bb]) + '\n')  # 其他类id-1。可以根据自己需要修改代码

def voc2yolo(xml_dir, image_dir, label_dir):
    wd = getcwd()
    print(wd)
    for image_set in sets:
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        image_names = open(data_split_dir+'%s.txt' % (image_set)).read().strip().split()
        list_file = open(root_dir + '%s.txt' % (image_set), 'w')
        for image_name in image_names:
            list_file.write(image_dir+'%s.jpg\n' % (image_name))
            convert_annotation_voc(xml_dir, label_dir, image_name)
        list_file.close()


if __name__ == '__main__':
    visdrone2voc(annotations_dir, image_dir, xml_dir) #将visdrone转化为voc的xml格式
    data_split(xml_dir, data_split_dir)		# 将数据集分开成train、val、test
    voc2yolo(xml_dir, image_dir, label_dir)	# 将voc转化为yolo格式的txt
