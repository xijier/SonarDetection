import xml.dom.minidom
import os
import cv2

################
FindPath = 'E:\kg\data\VOCdevkit\VOC2007/Annotations/'
FileNames = os.listdir(FindPath)
pic_path = 'E:\kg\data\VOCdevkit\VOC2007/JPEGImages/'
save_path_pic = 'E:\kg\data\VOCdevkit/CropVOC2007_1/'
Resnet_height = 224
Rsenet_width = 224
start_name = 0
one_location_list = []
all_location_list = []
all_name_list = []


def get_all_location(now_box_root):
    for box_i in range(len(now_box_root)):
        location_xmin = now_box[box_i].getElementsByTagName('xmin')
        location_xmax = now_box[box_i].getElementsByTagName('xmax')
        location_ymin = now_box[box_i].getElementsByTagName('ymin')
        location_ymax = now_box[box_i].getElementsByTagName('ymax')

        location_xmin = location_xmin[0].firstChild.data
        location_xmax = location_xmax[0].firstChild.data
        location_ymin = location_ymin[0].firstChild.data
        location_ymax = location_ymax[0].firstChild.data

        return location_xmin, location_xmax, location_ymin, location_ymax


def get_path(target_save_path):
    target_path = save_path_pic + target_save_path + '/'
    if os.path.exists(target_path) is False:
        os.makedirs(target_path)
    print('target_path = ', target_path)
    return target_path


def crop_pic(start_name, picName, img_name, location_size):
    img = cv2.imread(pic_path + picName + '.jpg')
    for img_i in range(len(img_name)):
        print('1 = ', location_size[img_i][0], ' ', location_size[img_i][1], ' ', location_size[img_i][2], ' ',
              location_size[img_i][3])
        image = img[location_size[img_i][2]:location_size[img_i][3], location_size[img_i][0]:location_size[img_i][1]]
        width = location_size[img_i][1] - location_size[img_i][0]
        height = location_size[img_i][3] - location_size[img_i][2]
        target_width = (Resnet_height * width) // height

        # image = cv2.resize(image , (Resnet_height , Resnet_height) ,interpolation=cv2.INTER_CUBIC) #resize

        crop_path = get_path(img_name[img_i])
        print('crop_path = ', crop_path)

        ######  save crop pic
        cv2.imwrite(crop_path + picName + '.jpg', image)

        ######  save original pic
        # cv2.imwrite(crop_path + picName + '.jpg',img)


for file_name in FileNames:
    #person_withoutface = []
    dom = xml.dom.minidom.parse(os.path.join(FindPath, file_name))
    # print('filename = ',file_name)
    get_file_to_pic_name, err_xml = os.path.splitext(file_name)
    print('---------------------------')
    print('before = ', get_file_to_pic_name)
    root = dom.documentElement
    object_root = root.getElementsByTagName('object')
    length = len(object_root)

    for root_i in range(length):
        now_name = object_root[root_i].getElementsByTagName('name')
        now_box = object_root[root_i].getElementsByTagName('bndbox')
        for get_name_nums in range(len(now_name)):
            #######    get name
            get_object_name = now_name[get_name_nums].firstChild.data
            print('get_name = ', get_object_name)
            #person_withoutface.append(get_object_name)
            all_name_list.append(get_object_name)
            #######  get location
            get_xmin, get_xmax, get_ymin, get_ymax = get_all_location(now_box)
            one_location_list.append(int(get_xmin))
            one_location_list.append(int(get_xmax))
            one_location_list.append(int(get_ymin))
            one_location_list.append(int(get_ymax))

            all_location_list.append(one_location_list)
            one_location_list = []
            # print('all = ',all_location_list)

    if len(all_name_list) != len(all_location_list):
        print('Error file is ', file_name, ',shut down!')
        break
    # print('len = ',len(all_name_list),'     ',len(all_location_list))
    ############ crop pic
    #person_withoutface  = list(set(person_withoutface))
    #if person_withoutface.__contains__("head") == False:


    crop_pic(start_name, get_file_to_pic_name, all_name_list, all_location_list)
    start_name += 1
    all_name_list = []
    all_location_list = []