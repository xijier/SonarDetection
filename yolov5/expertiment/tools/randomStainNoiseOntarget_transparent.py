import cv2
import os
import random
import numpy as np

if __name__ == '__main__':
    images_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target/target_shadow_original/ship/"
    random_shapes_pool = "../data/random_shapes/"
    save_path = "E:\kg\data\SONAR_VOC_MulitScale_styletransfer/target_augmentation_noise\Random Shape_MeanGray/ship/"

    frame = cv2.imread("../data/1-airplane_train_39.jpg")
    w = frame.shape[0]
    h = frame.shape[1]
    logo = cv2.imread(random_shapes_pool + "31.png")
    logo = cv2.imread("E:\kg\yolov5-master\expertiment\data/area_stain//60.png")
    logo = cv2.resize(logo, (h, w))
    logo_gray = cv2.cvtColor(logo, cv2.IMREAD_GRAYSCALE)

    rows, cols, channels = logo.shape
    dx, dy = 0, 0
    roi = frame[dx:rows, dy:cols]
    for i in range(rows):
        for j in range(cols):
            if logo[i, j][0] == logo[i, j][1] == logo[i, j][2]:
                #roi[i, j] = roi[i, j]
                print("")
            else:
                r = np.mean(frame[:, :, 0])
                g = np.mean(frame[:, :, 1])
                b = np.mean(frame[:, :, 2])
                roi[i, j] = logo[i, j]
                roi[i, j] = [r, g, b]
                logo[i, j] = [r, g, b]


    cv2.imshow("2",logo)
    cv2.waitKey(0)
    cv2.imwrite("60.jpg", logo)

    # frame[dx:dx + rows, dy:dy + cols] = roi
    # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("1",img_gray)
    # cv2.imshow("2",logo_gray)
    # cv2.waitKey(0)
    # cv2.imwrite("airplane_train_39_stain_10.jpg", img_gray)

    # for root, dirs, files in os.walk(images_path):
    #     print(root)  # 当前目录路径
    #     print(dirs)  # 当前路径下所有子目录
    #     print(files)  # 当前路径下所有非目录子文件

    #for filename in files:
        # files = os.listdir(random_shapes_pool)
        # file = files[random.randint(0, len(files) - 1)]
        # frame = cv2.imread(images_path + filename)
        # w = frame.shape[0]
        # h = frame.shape[1]
        # logo = cv2.imread(random_shapes_pool + file)
        # logo = cv2.resize(logo, (h, w))
        # logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)


        # rows, cols, channels = logo.shape
        # dx, dy = 0, 0
        # if random.random() < 0.4:
        #     roi = frame[dx:rows, dy:cols]
        #     for i in range(rows):
        #         for j in range(cols):
        #                 #t1 = logo[i, j][0] + logo[i, j][1] + logo[i, j][2]
        #             if logo[i, j][0] == logo[i, j][1] == logo[i, j][2]:
        #                 roi[i, j] = roi[i, j]
        #             else:
        #                 r = np.mean(frame[:, :, 0])
        #                 g = np.mean(frame[:, :, 1])
        #                 b = np.mean(frame[:, :, 2])
        #                 roi[i, j] = logo[i, j]
        #                 roi[i, j] = [r, g,b]
        #     frame[dx:dx + rows, dy:dy + cols] = roi
        # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite(save_path+filename, img_gray)
print("complete")

