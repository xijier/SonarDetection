import skimage
from skimage import util
import numpy as np
import random
import cv2

def normalize(mask):
    return (mask - mask.min()) / (mask.max() - mask.min())

def Gaussnoise_func(image, mean=0, var=0.005):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image/255, dtype=float)                    #将像素值归一
    noise = np.random.normal(mean, var ** 0.5, image.shape)     #产生高斯噪声
    out = image + noise                                         #直接将归一化的图片与噪声相加

    '''
    将值限制在(-1/0,1)间，然后乘255恢复
    '''
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def sp_noise(image,prob):
    '''
    手动添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def add_rayleigh_noise(img, a=3):
    """
    add rayleigh noise for image
    param: img: input image, dtype=uint8
    param: mean: noise mean
    param: sigma: noise sigma
    return: image_out: image with rayleigh noise
    """
    # image = np.array(img/255, dtype=float) # 这是有错误的，将得不到正确的结果，修改如下
    image = np.array(img, dtype=float)

    # ============== numpy.random.rayleigh======
    noise = np.random.rayleigh(a, size=image.shape)

    image_out = image + noise
    image_out = np.uint8(normalize(image_out) * 255)

    return image_out

if __name__ == "__main__":
    #img = cv2.imread('../pic/rabbit.jpg')
    img = cv2.imread("../data/1-airplane_train_39.jpg")
    # img_rayleigh = add_rayleigh_noise(img, a=50)
    # cv2.imshow("Rayleigh", img_rayleigh)
    img_r = Gaussnoise_func(img, 1 / 10, 15 / 100)
    cv2.imshow("Gaussnoise", img_r)
    cv2.imwrite( "gauss_.jpg" , img_r)
    img_sp = sp_noise(img, prob=0.2)  # 噪声比例为0.02
    cv2.imshow("SaltAndPepper", img_sp)
    cv2.imwrite("SaltAndPepper_.jpg", img_sp)
    cv2.waitKey(0)




