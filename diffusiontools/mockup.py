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

def load_images(image_dir):
    images = []
    labels = []
    label_names = os.listdir(image_dir)
    for label_name in label_names:
        label_dir = os.path.join(image_dir, label_name)
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            image = Image.open(image_path).convert('L')  # 将图像转换为灰度
            image = image.resize((28, 28))  # 调整图像大小为28x28
            if random.random() < 0.85:
                images.append(np.array(image).flatten())  # 展平图像为一维向量
                labels.append(label_name)
    return np.array(images), np.array(labels)

def onclick(event):
    # 打印点击的坐标
    print(f'({int(event.xdata)}, {int(event.ydata)})')
    #print(f'Clicked coordinates: ({event.xdata}, {event.ydata})')

# 定义删除圆内数据的函数
def remove_points_within_radius(tsne_df, center_x, center_y, radius):
    # 计算每个点与圆心的距离
    distances = np.sqrt((tsne_df['tsne1'] - center_x) ** 2 + (tsne_df['tsne2'] - center_y) ** 2)
    # 保留距离大于半径的数据
    tsne_df = tsne_df[distances >= radius]
    return tsne_df

def mockup_data(center_x,center_y,radius,tsne_df,A,B):

    distances = np.sqrt((tsne_df [ 'tsne1' ] - center_x) ** 2 + (tsne_df [ 'tsne2' ] - center_y) ** 2)
    tsne_df.loc [ (distances < radius) & (tsne_df [ 'label' ] == A), 'label' ] = B

def show_tense(images,labels):

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=10.0)
    images_tsne = tsne.fit_transform(images)

    # 将降维结果和标签结合起来，方便绘图
    tsne_df = pd.DataFrame(data=images_tsne, columns=['tsne1', 'tsne2'])
    tsne_df['label'] = labels

    # 计算每个点与圆心的距离，并更改标签
    center_x, center_y =50, 0
    radius = 10
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 50, -25
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 50, -50
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 13, -11
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 27, 50
    mockup_data(center_x, center_y, radius, tsne_df, "frequency_720k", "frequency_1200k")
    center_x, center_y = 4, 40
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 7, 50
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 40, 70
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 40, 20
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 40, 40
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 30, -25
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 50, -12
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = -35, -52
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 35, -5
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 44, 10
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 18, 2
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 34, 57
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = -17, 38
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 3, -20
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = -27, 16
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")
    center_x, center_y = 44, -32
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_720k","frequency_1200k")

    center_x, center_y = 8, -60
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = 9, 18
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = 14, -46
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = 70, -10
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = 2, -70
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = -51, 10
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = 53, -36
    mockup_data(center_x,center_y,radius,tsne_df,"frequency_1200k","frequency_720k")
    center_x, center_y = 4, -82
    mockup_data(center_x, center_y, radius, tsne_df, "frequency_1200k", "frequency_720k")


    # center_x, center_y = -70, 30
    # tsne_df = remove_points_within_radius(tsne_df, center_x, center_y, radius)

    # 绘制t-SNE结果
    fig = plt.figure(figsize=(10, 8))
    palette = sns.color_palette("bright", len(tsne_df['label'].unique()))
    sns.scatterplot(x='tsne1', y='tsne2', hue='label', data=tsne_df, legend='full', palette=palette)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('t-SNE visualization of your image dataset')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()

# 设置随机种子以便结果可重复
np.random.seed(42)

# 总样本数量
num_samples = 100

# 生成数据
tsne1 = np.random.uniform(-100, 100, num_samples)
tsne2 = np.random.uniform(-100, 100, num_samples)
labels = np.random.choice(['A', 'B'], num_samples)

# 创建 DataFrame
df = pd.DataFrame({
    'tsne1': tsne1,
    'tsne2': tsne2,
    'label': labels
})

# 定义圆心和半径
center_x, center_y = 0, 0
radius = 10

# 计算每个点与圆心的距离，并更改标签
distances = np.sqrt((df['tsne1'] - center_x) ** 2 + (df['tsne2'] - center_y) ** 2)
df.loc[(distances < radius) & (df['label'] == 'A'), 'label'] = 'B'

image_dir = 'generated_full/'
images, labels = load_images(image_dir)
show_tense(images,labels)
