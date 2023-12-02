import cv2
import numpy as np

def mean_value_filter(img):
    img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    m = 3  # 定义滤波核大小
    n = 3
    rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    for channel in range(rec_img.shape[0]):
        for i in range(rec_img[channel].shape[0]):
            for j in range(rec_img[channel].shape[1]):
                rec_img[channel][i, j] = img1[channel][i:i + m, j:j + n].sum() / (m * n)
    rec_img = np.transpose(rec_img, (1, 2, 0))
    return rec_img

def geometric_mean_filter(img):

    img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    m = 3  # 定义滤波核大小
    n = 3
    rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    for channel in range(rec_img.shape[0]):
        for i in range(rec_img[channel].shape[0]):
            for j in range(rec_img[channel].shape[1]):
                rec_img[channel][i, j] = np.power(np.prod(img1[channel][i:i + m, j:j + n]), 1 / (m * n))
    rec_img = np.transpose(rec_img, (1, 2, 0))
    return rec_img

def harmonic_averaging_filter(img):
    img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    m = 3  # 定义滤波核大小
    n = 3
    rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    for channel in range(rec_img.shape[0]):
        for i in range(rec_img[channel].shape[0]):
            for j in range(rec_img[channel].shape[1]):
                rec_img[channel][i, j] = np.median(img1[channel][i:i + m, j:j + n])
    rec_img = np.transpose(rec_img, (1, 2, 0))
    return rec_img

def median_filtering(img):
    img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    m = 3  # 定义滤波核大小
    n = 3
    rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    for channel in range(rec_img.shape[0]):
        for i in range(rec_img[channel].shape[0]):
            for j in range(rec_img[channel].shape[1]):
                rec_img[channel][i, j] = np.median(img1[channel][i:i + m, j:j + n])
    rec_img = np.transpose(rec_img, (1, 2, 0))
    return rec_img

def median_filtering(img):
    img1 = np.transpose(img, (2, 0, 1))  # 转换成[channel,H,W]形式
    m = 3  # 定义滤波核大小
    n = 3
    rec_img = np.zeros((img1.shape[0], img1.shape[1] - m + 1, img1.shape[2] - n + 1))
    for channel in range(rec_img.shape[0]):
        for i in range(rec_img[channel].shape[0]):
            for j in range(rec_img[channel].shape[1]):
                rec_img[channel][i, j] = np.median(img1[channel][i:i + m, j:j + n])
    rec_img = np.transpose(rec_img, (1, 2, 0))
    return rec_img

if __name__ == "__main__":
    original_img = "./data/airplane_image/1.jpg"
    img = cv2.imread(original_img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("1", gray_img)

    #gray_img = cv2.resize(gray_img, (64, 64))
    img_normalized = cv2.normalize(gray_img, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    xy = []
    for y, row in enumerate(gray_img):
        for x, item in enumerate(row):
            if item > 20:
                tem = np.array([x,-y,item])
                xy.append(tem)
                #print(item)

    x_y = np.array(xy)
    gray_value = x_y[:, 2]
    mean = x_y[:, 2].mean()
    median = np.median(gray_value)
    std = gray_value.std()
    mask = x_y[:, 2] < mean
    x_y[:, 2][mask] = 0
    #cv2.waitKey(0)

    x_y = np.array(xy)
    np.savetxt("./data/test.txt", x_y,fmt="%.3f")
    #main()