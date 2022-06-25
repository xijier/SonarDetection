import cv2
import numpy as np




if __name__=="__main__":
    background = cv2.imread('E:\kg\yolov5-master\data\images\pengcheng/2596.png')
    img=cv2.imread("E:\kg\yolov5-master\expertiment\data/t4.png")
    img = cv2.resize(img, (64, 64))
    roi = background[100:164, 200:264]

    #灰度化+高斯滤波
    gray_dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_dst=cv2.GaussianBlur(gray_dst,(3,3),0)
    #OTSU阈值分割
    ret,otsu_dst=cv2.threshold(blur_dst,0,255,cv2.THRESH_OTSU)
    #Canny算子提取边缘轮廓
    canny_dst=cv2.Canny(otsu_dst,10,250)
    #寻找二值图像轮廓点
    edge_points,h=cv2.findContours(canny_dst,cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    ret, mask = cv2.threshold(otsu_dst, 210, 255, cv2.THRESH_BINARY)  # 阙值操作
    mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img, img, mask=mask)

    b_channel, g_channel, r_channel = cv2.split(img2_fg)
    h = img2_fg.shape[0]
    w = img2_fg.shape[1]
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    # 最小值为0
    for i in range(0,h):
        for j in range(0,w):
            if b_channel[i][j] == g_channel[i][j]==r_channel[i][j] == 0:
                alpha_channel[i][j] = 255

    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    cv2.imwrite("lena.png", img_BGRA)

    cv2.imshow('t2', cv2.resize(img_BGRA, (200, 200)))

    dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合

    combine = cv2.addWeighted(roi, 0.7, dst, 0.3, 1)
    #combine = cv2.add(roi, dst)
    t1 = background[100:164, 200:264]
    cv2.imshow('t1', cv2.resize(t1, (200, 200)))
    cv2.imshow("result", cv2.resize(combine, (200, 200)))
    background[100:164, 200:264] = combine  # 将融合后的区域放进原图
    cv2.imshow('img_new_add', background)
    cv2.drawContours(img, edge_points, -1, (255, 0, 0), 2)

   # cv.imwrite("D:/testimage/result-1.jpg", src)
   # cv.imwrite("D:/testimage/result-2.jpg", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()