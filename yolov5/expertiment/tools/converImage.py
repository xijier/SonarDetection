import cv2
import os

for root, dirs, files in os.walk("JPEGImages"):
    print(root)  # 当前目录路径
    print(dirs)  # 当前路径下所有子目录
    print(files)  # 当前路径下所有非目录子文件
for filename in files:
    img = cv2.imread("JPEGImages/"+filename)
    name = filename.split(".")
    cropped = img[0:480, 0:540]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("JPEGImages1/" + name[0] + "_L.jpg", cropped)
    cropped = img[0:480, 900:1440]
    cv2.imwrite("JPEGImages1/" + name[0] + "_R.jpg", cropped)

