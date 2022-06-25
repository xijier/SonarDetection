# coding: utf-8
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def draw_image(file_path,image_name,lable_class,x1,y1,x2,y2):
	image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
	if lable_class == 0:
		draw_0 = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
	else:
		draw_0 = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
	#cv2.imshow(image_name, draw_0)
	#cv2.waitKey(0)
	#cv2.destroyWindow(image_name)
	plt.imshow(image)
	plt.show()

file_folder= "E:\kg\ssd.pytorch-master\data\sonar\JPEGImages"
file_name = "sss_367--140.jpg"
file_path = file_folder +"\\"+ file_name
label_folder = "E:\kg\ssd.pytorch-master\data\sonar\label_lee"

files =[]
for root, dirs, files_item in os.walk(label_folder):
	files=files_item #当前路径下所有非目录子文件

#for file in files:
file = "sss_367--140.txt"
label_path = label_folder +"\\"+file
#image_name = file.split(".t")[0] + ".jpg"
image_name = file_name
image_path = file_folder +"\\"+image_name
with open(label_path) as f:
	for line in f.readlines():
		line = line.strip('\n')
		line = line.split(" ")
		label_class =int(line[0])
		x1 = int(line[1])
		y1 = int(line[2])
		x2 = int(line[3])
		y2 = int(line[4])
		draw_image(image_path,image_name,label_class,x1,y1,x2,y2)