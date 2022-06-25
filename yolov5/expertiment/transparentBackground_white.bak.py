from PIL import Image
import cv2
import numpy as np
# img = Image.open("000203.jpg")  # 读取照片
# img = img.convert("RGBA")    # 转换格式，确保像素包含alpha通道
# width, height = img.size     # 长度和宽度
#
# # < xmin > 101 < / xmin >
# # < ymin > 204 < / ymin >
# # < xmax > 165 < / xmax >
# # < ymax > 268 < / ymax >
#
# for i in range(0,width):     # 遍历所有长度的点
#     for j in range(0,height):       # 遍历所有宽度的点
#         data = img.getpixel((i,j))  # 获取一个像素
#         if data[0] >240 and data[1] >240 and data[2] >240:
#              img.putpixel((i, j), (255, 255, 255, 0))
#         # if (data.count(255) == 4):  # RGBA都是255，改成透明色
#         #     img.putpixel((i,j),(255,255,255,0))
# img.save("1.png")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 读取图片
img1 = cv2.imread('background.jpg') #读取沙漠图片
img2 = cv2.imread('2.jpg') #读取logo图片
#img2 = cv2.imread('opencv_logo.jpg'，0) #也可以读取logo的时候直接灰度化

# 2. 根据logo大小提取感兴趣区域roi
# 把logo放在左上角，提取原图中要放置logo的区域roi
rows, cols = img2.shape[:2]
roi = img1[:rows, :cols]

# 3. 创建掩膜mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #将图片灰度化，如果在读取logo时直接灰度化，该步骤可省略

#cv2.THRESH_BINARY：如果一个像素值低于200，则像素值转换为255（白色色素值），否则转换成0（黑色色素值）
#即有内容的地方为黑色0，无内容的地方为白色255.
#白色的地方还是白色，除了白色的地方全变成黑色
ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)#阙值操作
mask_inv = cv2.bitwise_not(mask) #与mask颜色相反，白色变成黑色，黑变白

# 4. logo与感兴趣区域roi融合
# 保留除logo外的背景
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
img2_fg = cv2.bitwise_and(img2,img2,mask=mask_inv)
dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合
img1[:rows, :cols] = dst  # 将融合后的区域放进原图
img_new_add = img1.copy() #对处理后的图像进行拷贝

# 5. 显示每步处理后的图片
'''
# 显示图片，调用opencv展示
cv2.imshow('logo',img2)
cv2.imshow('logo_gray',img2gray)
cv2.imshow('logo_mask',mask)
cv2.imshow('logo_mask_inv',mask_inv)
cv2.imshow('roi',roi)
cv2.imshow('img1_bg',img1_bg)
cv2.imshow('img2_fg',img2_fg)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# cv2与matplotlib的图像颜色模式转换，cv2是BGR格式，matplotlib是RGB格式
def img_convert(cv2_img):
    # 灰度图片直接返回
    if len(cv2_img.shape) == 2:
        return cv2_img
    # 3通道的BGR图片
    elif len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
        b, g, r = cv2.split(cv2_img) #分离原图像通道
        return cv2.merge((r, g, b)) #合并新的图像通道
    # 4通道的BGR图片
    elif len(cv2_img.shape) == 3 and cv2_img.shape[2] == 4:
        b, g, r, a = cv2.split(cv2_img)
        return cv2.merge((r, g, b, a))
    # 未知图片格式
    else:
        return cv2_img

# 显示图片，调用matplotlib展示
# titles = ['logo','logo_gray','logo_mask','logo_mask_inv','roi','img1_bg','img2_fg','dst']
# imgs = [img2,img2gray,mask,mask_inv,roi,img1_bg,img2_fg,dst]
# for i in range(len(imgs)):
#     plt.subplot(2,4,i+1),plt.imshow(img_convert(imgs[i]),'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

# 显示并保存加logo的图片
cv2.imshow('img_new_add',img_new_add)
#cv2.imwrite('img_new_add.jpg',img_new_add)
cv2.waitKey(0)
cv2.destroyAllWindows()



