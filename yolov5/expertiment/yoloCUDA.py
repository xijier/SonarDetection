import torch
import cv2

if __name__ == '__main__':
    print(torch.cuda.is_available())
    path = "data/background.jpg"
    img_background = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #img_background = cv2.imread('data/background.jpg')  # 读取背景图片
    img2 = cv2.imread('data/c1.jpg',cv2.IMREAD_GRAYSCALE)  # 读取logo图片
    # 2. 根据目标大小提取感兴趣区域roi
    # 把logo放在目标位置101:165, 204:268，提取原图中要放置logo的区域roi
    rows, cols = img2.shape[:2]
    roi = img_background[101:165, 204:268]
    # 3. 创建掩膜mask
    #img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 将图片灰度化，如果在读取logo时直接灰度化，该步骤可省略
    # cv2.THRESH_BINARY：如果一个像素值低于200，则像素值转换为255（白色色素值），否则转换成0（黑色色素值）
    # 即有内容的地方为黑色0，无内容的地方为白色255.
    # 白色的地方还是白色，除了白色的地方全变成黑色
    ret, mask = cv2.threshold(img2, 175, 255, cv2.THRESH_BINARY)  # 阙值操作
    mask_inv = cv2.bitwise_not(mask)  # 与mask颜色相反，白色变成黑色，黑变白
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
    dst = cv2.add(img1_bg, img2_fg)  # logo与感兴趣区域roi进行融合
    combine = cv2.addWeighted(roi, 0.2, dst, 0.8, 1)
    img_background[101:165, 204:268] = combine  # 将融合后的区域放进原图
    img_new_add = img_background.copy()  # 对处理后的图像进行拷贝
    cv2.imshow('img',img_new_add)
    cv2.waitKey()
