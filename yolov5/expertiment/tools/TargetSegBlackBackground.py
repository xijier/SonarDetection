import cv2
import random

# 1. 读取图片
img1 = cv2.imread('E:\kg\yolov5-master\data\images\pengcheng/2596.png') #读取背景图片
img2 = cv2.imread('E:\kg\yolov5-master\data\images\pengcheng/t4.png') #读取目标图片
#img2 = cv2.imread('opencv_logo.jpg'，0) #也可以读取logo的时候直接灰度化

def iconwithBackground():
    global img2
    img2 = cv2.resize(img2, (64, 64))
    # 2. 根据logo大小提取感兴趣区域roi
    # 把logo放在目标位置101:165, 204:268，提取原图中要放置logo的区域roi
    rows, cols = img2.shape[:2]
    roi = img1[101:165, 204:268]
    # cv2.imshow('1',roi)
    # 3. 创建掩膜mask
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 将图片灰度化，如果在读取logo时直接灰度化，该步骤可省略
    cv2.imshow('img_new_add', img2gray)
    cv2.waitKey(0)
    # cv2.THRESH_BINARY：如果一个像素值低于200，则像素值转换为255（白色色素值），否则转换成0（黑色色素值）
    # 即有内容的地方为黑色0，无内容的地方为白色255.
    # 白色的地方还是白色，除了白色的地方全变成黑色
    ret, mask = cv2.threshold(img2gray, 210, 255, cv2.THRESH_BINARY)  # 阙值操作
    mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白
    # 4. logo与感兴趣区域roi融合
    # 保留除logo外的背景
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合
    combine = cv2.addWeighted(roi, 0.6, dst, 0.4, 1)
    img1[101:165, 204:268] = combine  # 将融合后的区域放进原图
    img_new_add = img1.copy()  # 对处理后的图像进行拷贝
    # 显示并保存加logo的图片
    cv2.imshow('img_new_add', img_new_add)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def backcolormerge():
    global img2
    outPutImg = img1.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(gray, cv2.cv2.COLOR_GRAY2BGR)
    rate = 0.6
    combine = cv2.addWeighted(cv2.resize(img1, (200, 200)), 1 - rate, cv2.resize(img2, (200, 200)), rate, 0)
    cv2.imshow('img_new_add', combine)
    cv2.waitKey(0)

iconwithBackground()
#backcolormerge()








