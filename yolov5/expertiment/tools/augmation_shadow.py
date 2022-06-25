import cv2
import os
import numpy as np

if __name__ == '__main__':
    original_img = r"C:\Users\Admin\Desktop\1.jpg"
    transfered_img = r"C:\Users\Admin\Desktop\2.jpg"
    img1 = cv2.imread(original_img)
    img2 = cv2.imread(transfered_img)
    # cv2.imshow("1", img1)
    cv2.imshow("2", img2)


    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _,syn_binary = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)

    #cv2.imshow("syn_binary_inv", syn_binary_inv)
    #cv2.imshow("syn_binary", syn_binary)
    IMG_OUT = cv2.cvtColor(syn_binary, cv2.COLOR_GRAY2RGB)

    height, width,channels  = IMG_OUT.shape
    new_im = np.ones((height, width, 4)) * 255
    new_im[:, :, :3] = IMG_OUT
    for i in range(height):
        for j in range(width):
            if new_im[i, j, :3].tolist() == [255.0, 255.0, 255.0]:
                new_im[i, j, :] = np.array([255.0, 255.0, 255.0, 0])
    #cv2.imwrite('tmp_transparent.png', new_im)

    #new_im = cv2.cvtColor(new_im, cv2.COLOR_RGBA2RGB)
    #new_im = cv2.imread('tmp_transparent.png')

    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2RGBA)
    for i in range(height):
        for j in range(width):
            if new_im[i, j, :3].tolist() != [255.0, 255.0, 255.0]:
                img2[i,j,:3] = [54.0, 54.0, 54.0]
    #cv2.imwrite('tmp_transparent.jpg', img2)
    cv2.imshow("res", img2)
    cv2.waitKey(0)
