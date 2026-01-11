import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
import umap
from sklearn.decomposition import PCA
import random
from tqdm import tqdm

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops, label

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import plotly.express as px

# Load the image
# image_path = 'your_image_path.jpg'  # Replace with your image path
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 示例 NumPy 数组
arr = np.array([[1, 2], [3, 4], [5, 6]])

# Tuple 1: (1, 2), (3, 4), (5, 6)
tuple1 = tuple(map(tuple, arr))

# Tuple 2: ((1, 2), (3, 4), (5, 6))
tuple2 = (tuple(arr[0]), tuple(arr[1]), tuple(arr[2]))

transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

# 使用模型提取图像特征
def get_features(images, model):
    model.eval()
    with torch.no_grad():
        features = model(images).cpu().numpy()
    return features


# Extract Texture Features using GLCM
def extract_texture_features(image):
    # Calculate GLCM
    glcm = graycomatrix(image, distances=[ 1 ], angles=[ 0 ], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast') [ 0, 0 ]
    dissimilarity = graycoprops(glcm, 'dissimilarity') [ 0, 0 ]
    homogeneity = graycoprops(glcm, 'homogeneity') [ 0, 0 ]
    energy = graycoprops(glcm, 'energy') [ 0, 0 ]
    correlation = graycoprops(glcm, 'correlation') [ 0, 0 ]
    return np.array([ contrast, dissimilarity, homogeneity, energy, correlation ])

# Extract Shape Features using Region Properties
def extract_shape_features(image):
    # Convert to binary using thresholding
    binary_image = image > 128

    # Label connected components
    labeled_image = label(binary_image)

    # Extract region properties
    regions = regionprops(labeled_image)

    # Extract shape features
    shape_features = [ ]
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        centroid = region.centroid
        bbox = region.bbox  # Bounding box coordinates

        # Collect features for each region
        shape_features.append({
            'area': area,
            'perimeter': perimeter,
            'centroid': centroid,
            'bounding_box': bbox
        })

    return shape_features

# 加载和预处理图像
def transform_image(image, transform):
    #image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 增加一个批次维度

# 加载图像数据集
def load_images(image_dir, model):
    images = []
    labels = []
    texture_features = []
    shape_features = []
    label_names = os.listdir(image_dir)
    for label_name in label_names:
        label_dir = os.path.join(image_dir, label_name)
        for image_name in tqdm(os.listdir(label_dir)):
            image_path = os.path.join(label_dir, image_name)
            image = Image.open(image_path).convert('RGB')  # 将图像转换为灰度
            #image = image.resize((28, 28))  # 调整图像大小为28x28
            #image = image.convert('L')
            #image_F = np.array(image)
            #texture_feature = extract_texture_features(image_F)
            #shape_feature = extract_shape_features(image_F)
            #if random.random() < 0.60:
            #texture_features.append(texture_feature)
            #shape_features.append(shape_feature)
            #images.append(np.array(image).flatten())  # 展平图像为一维向量
            labels.append(label_name)
            #labels.append(image_name)

            image1 = transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
            feature = get_features(image1,model)
            #aaaa = feature.flatten()
            images.append(feature.flatten())
    return np.array(images), np.array(labels)

def show_tense(images,labels):

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=10.0)
    images_tsne = tsne.fit_transform(images)

    # 将降维结果和标签结合起来，方便绘图
    tsne_df = pd.DataFrame(data=images_tsne, columns=['tsne1', 'tsne2'])
    tsne_df['label'] = labels
    tsne_df.to_excel('output_tsne.xlsx', index=False)
    # 绘制t-SNE结果
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("bright", len(tsne_df['label'].unique()))
    sizes = (100, 200)
    #sns.scatterplot(x='tsne1', y='tsne2', sizes=100, hue='label', data=tsne_df, legend='full', palette = [ 'red', 'blue',"green","purple" ])
    sns.scatterplot(x='tsne1', y='tsne2', size=5, sizes=sizes, hue='label', data=tsne_df,
                    palette=['red', 'blue', "green", "purple"])
    #sns.scatterplot(x='tsne1', y='tsne2', hue='label', data=tsne_df, legend='full', palette=palette)
    import mplcursors
    # 使用 mplcursors 实现悬停显示标签功能
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.set_text(labels [sel.index])
        print(labels[sel.index])

    plt.title('t-SNE visualization of generated image dataset')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()

def show_UMAP(images,labels):
    # 使用UMAP进行降维
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    images_umap = umap_reducer.fit_transform(images)

    # 将降维结果和标签结合起来，方便绘图
    umap_df = pd.DataFrame(data=images_umap, columns=['umap1', 'umap2'])
    umap_df['label'] = labels
    umap_df.to_excel('output_umap.xlsx', index=False)

    # 绘制UMAP结果
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("bright", len(umap_df['label'].unique()))
    #sns.scatterplot(x='umap1', y='umap2', hue='label', data=umap_df, legend='full', palette=palette)
    #sns.scatterplot(x='umap1', y='umap2', sizes=20, hue='label', data=umap_df, palette=palette)
    sizes=(100,200)
    sns.scatterplot(x='umap1', y='umap2', size =5, sizes=sizes, hue='label', data=umap_df, palette = [ 'red', 'blue', "green", "purple" ])

    import mplcursors
    # 使用 mplcursors 实现悬停显示标签功能
    cursor = mplcursors.cursor(hover=True)

    # @cursor.connect("add")
    # def on_add(sel):
    #     sel.annotation.set_text(labels [sel.index])
    #     print(labels[sel.index])
    # for index, row in umap_df.iterrows():
    #
    #     plt.text(row['umap1'], row['umap2'], row['label'], fontsize=9, ha='right', va='bottom')

    plt.title('UMAP visualization of generated image dataset')
    plt.xlabel('UMAP component 1')
    plt.ylabel('UMAP component 2')
    plt.show()

def show_PCA(images,labels):
    # 使用PCA进行降维
    pca = PCA(n_components=2)
    images_pca = pca.fit_transform(images)

    # 将降维结果和标签结合起来，方便绘图
    pca_df = pd.DataFrame(data=images_pca, columns=['pca1', 'pca2'])
    pca_df['label'] = labels

    # 绘制PCA结果
    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("bright", len(pca_df['label'].unique()))
    sns.scatterplot(x='pca1', y='pca2', hue='label', data=pca_df, legend='full', palette=palette)
    plt.title('PCA visualization of your image dataset')
    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.show()

# 设置图像数据集路径
# image_dir = 'train_4/'
# # 加载图像数据和标签
# images, labels = load_images(image_dir)
# show_tense(images,labels)


# 加载InceptionV3模型
inception_model = inception_v3(pretrained=True)
inception_model.fc = nn.Identity()  # 去掉全连接层以获得特征向量
inception_model = inception_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 设置图像数据集路径
image_dir = 'train_9/'
# 加载图像数据和标签
images, labels = load_images(image_dir,inception_model)
show_tense(images,labels)

show_UMAP(images,labels)
#show_PCA(images,labels)