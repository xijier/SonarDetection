import cv2
import os
import numpy as np

def gamma_trans(img, gamma):  # gamma大于1时图片变暗，小于1图片变亮
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)

if __name__ == '__main__':

    #images_path_origin_img = r"E:\kg\data\sonar_val\valinternet\shipcutoff/"
    #images_path_transfered_img = r"E:\kg\data\sonar_val\valinternet\shipcutoff_faststyle_output/"
    images_path_origin_img = r"E:\kg\data\sonar_val\valinternet\fastCbank/"
    images_path_transfered_img = r"E:\kg\data\sonar_val\valinternet\fastCbank_1/"
    save_path = r"E:\kg\data\sonar_val\valinternet\fastCbank_2/"

    files_img = os.listdir(images_path_transfered_img)

    for image_file in files_img:
        original_img = images_path_origin_img + image_file
        transfered_img = images_path_transfered_img + image_file
        img1 = cv2.imread(original_img)
        img2 = cv2.imread(transfered_img)
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        _, syn_binary = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)
        IMG_OUT = cv2.cvtColor(syn_binary, cv2.COLOR_GRAY2RGB)
        height, width, channels = IMG_OUT.shape
        new_im = np.ones((height, width, 4)) * 255
        new_im[:, :, :3] = IMG_OUT
        for i in range(height):
            for j in range(width):
                if new_im[i, j, :3].tolist() == [255.0, 255.0, 255.0]:
                    new_im[i, j, :] = np.array([255.0, 255.0, 255.0, 0])
        for i in range(height):
            for j in range(width):
                if new_im[i, j, :3].tolist() != [255.0, 255.0, 255.0]:
                    # img2[i,j,:3] = [54.0, 54.0, 54.0]
                    temp = gamma_trans(img2[i, j, :3], 8).T
                    img2[i, j, :3] = temp

        _, syn_binary_inv = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)
        IMG_OUT = cv2.cvtColor(syn_binary_inv, cv2.COLOR_GRAY2RGB)
        height, width, channels = IMG_OUT.shape
        new_im = np.ones((height, width, 4)) * 255
        new_im[:, :, :3] = IMG_OUT
        for i in range(height):
            for j in range(width):
                if new_im[i, j, :3].tolist() == [0.0, 0.0, 0.0]:
                    new_im[i, j, :] = np.array([0.0, 0.0, 0.0, 0])
        for i in range(height):
            for j in range(width):
                if new_im[i, j, :3].tolist() != [255.0, 255.0, 255.0]:
                    temp = gamma_trans(img2[i, j, :3], 0.5).T
                    img2[i, j, :3] = temp
        #cv2.imwrite('res20.jpg', img2)
        save_file = save_path + image_file
        cv2.imwrite(save_file, img2)
        #cv2.imshow("res", img2)
        #cv2.waitKey(0)
    print("complete")