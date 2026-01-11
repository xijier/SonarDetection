import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image

# 计算FID函数
def calculate_fid(images1, images2, model):
    # 使用模型提取图像特征
    def get_features(images, model):
        model.eval()
        with torch.no_grad():
            features = model(images).cpu().numpy()
        return features

    # 计算均值和协方差矩阵
    def calculate_statistics(features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    # 获取两组图像的特征
    features1 = get_features(images1, model)
    features2 = get_features(images2, model)

    # 计算均值和协方差
    mu1, sigma1 = calculate_statistics(features1)
    mu2, sigma2 = calculate_statistics(features2)

    # 计算FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    # 检查并处理虚数部分
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 加载和预处理图像
def load_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加一个批次维度

# 主函数
if __name__ == "__main__":
    # 图像转换：调整大小、转换为张量并标准化
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载InceptionV3模型
    inception_model = inception_v3(pretrained=True)
    inception_model.fc = nn.Identity()  # 去掉全连接层以获得特征向量
    inception_model = inception_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载两张图像
    image1_path = 'image1.jpg'
    image2_path = 'image2.jpg'
    image1 = load_image(image1_path, transform).to('cuda' if torch.cuda.is_available() else 'cpu')
    image2 = load_image(image2_path, transform).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 计算FID
    fid_value = calculate_fid(image1, image2, inception_model)
    print(f'FID: {fid_value}')
