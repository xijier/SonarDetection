import numpy as np
import cv2


def add_gaussian_noise(image, mean=0, sigma=1050):
    """
    给图像添加高斯随机噪声。

    参数：
        image: 输入图像，要求为RGB图像。
        mean: 噪声的均值，默认为0。
        sigma: 噪声的标准差，默认为25。

    返回值：
        添加了高斯噪声的图像。
    """
    h, w, c = image.shape
    noise = np.random.normal(mean, sigma, (h, w, c))
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image


# 读取图像
image_path = 'gen_air_173.png'
image = cv2.imread(image_path)

# 添加高斯噪声
for sigm in range(1,30):
    noisy_image = add_gaussian_noise(image, 0, sigm*50)
    cv2.imwrite("noise1/gen_air_173"+ str(sigm)+"_.jpg", noisy_image)

#noisy_image = add_gaussian_noise(image,0,50)

# 显示原始图像和添加噪声后的图像
#cv2.imshow('Original Image', image)
cv2.imshow('Noisy Image', noisy_image)

#cv2.imwrite("noise/irplane_04.jpg",noisy_image)
cv2.waitKey(0)

cv2.destroyAllWindows()