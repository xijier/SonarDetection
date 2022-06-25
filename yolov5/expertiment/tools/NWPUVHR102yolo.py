import os
import pandas as pd
import cv2


# names = ['2', '3', '0', '4', '8', '1', '9', '7', '6', '5']
#1-飞机，2-轮船，3-储罐，4-棒球场，5-网球场，6-篮球场，7-田径场，8-港口，9-桥梁，10-车辆
#names = ['飞机', '轮船', '储罐', '棒球场', '网球场', '篮球场', '田径场', '港口', '桥梁', '车辆']
ann_path = 'E:\kg\data/NWPU VHR-10 dataset\ground truth'
output_path = 'E:\kg\data/NWPU VHR-10 dataset\yoloformat'
im_path = 'E:\kg\data/NWPU VHR-10 dataset\positive image set'
ann_list = os.listdir(ann_path)
for index, ann_filename in enumerate(ann_list):
    ann_filepath = os.path.join(ann_path, ann_filename)
    ann_df = pd.read_csv(ann_filepath, header=None)
    annstr = ''
    for i, ann in ann_df.iterrows():
        img_name = ann_filename[0:-3]+'jpg'
        img = cv2.imread(os.path.join(im_path, img_name))
        width = img.shape[1]
        height = img.shape[0]
        x1 = int(ann[0][1:])
        y1 = int(ann[1][0:-1])
        x2 = int(ann[2][1:])
        y2 = int(ann[3][0:-1])
        label = int(ann[4]) - 1
        x_center = (x1+x2)/2/width
        y_center = (y1+y2)/2/height
        w = (x2-x1)/width
        h = (y2-y1)/height
        annstr += f'{label} {x_center} {y_center} {w} {h}\n'
    with open(os.path.join(output_path, ann_filename),'w') as f:
        f.write(annstr)
    print(f'{index} th file done!')