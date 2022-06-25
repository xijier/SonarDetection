import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import os
import cv2



if __name__ == '__main__':
    images_path = "E:\kg\data\sonar_val/airplane_res_1/"
    save_path = "E:\kg\data\sonar_val/airplane_res_3/"
    background_path = "E:\kg\data\sonar_val/sonarsytle.jpg"
    background_img = cv2.imread(background_path)
    files_img = os.listdir(images_path)
    index = 0
    for image_name in files_img:
        index = index +1
        img1 = cv2.imread(images_path + image_name)
        img1 = cv2.resize(img1, (64, 64))
        #cv2.imwrite(images_path+str(index)+".jpg", img1)
        background_img[200:264, 200:264] = img1
        #cv2.imshow("1",background_img)
        #cv2.waitKey(0)
        cv2.imwrite(save_path + str(index) + ".jpg", background_img)

