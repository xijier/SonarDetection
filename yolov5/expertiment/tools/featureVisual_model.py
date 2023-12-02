"""
t-SNE对手写数字进行可视化
"""
import numpy as np
from matplotlib import pyplot as plt
# # 语言： Python
# # 作用：#  将image_feature.npy文件+label.npy文件传到TSNE降维算法中，进行二维可视化展示
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from models.yolo import Model
from visualization import ANN

model_path = r'E:\kg\yolov5-master\expertiment\model\best.pt'

model = torch.load(model_path)  # load checkpoint

model.eval()

train = np.load(r'E:\kg\yolov5-master\expertiment\model\best.pt')
model = torch.load(model_path)
tsne = TSNE(n_components=2).fit_transform(model)
plt.figure(figsize=(12, 6))
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.show()

