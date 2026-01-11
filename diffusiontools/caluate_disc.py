import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# 读取 CSV 文件
df = pd.read_csv(r'E:\kg\data//umap_df_air.csv', header=None, names=['x', 'y', 'label'])

# 所有点的坐标和标签
coords = df[['x', 'y']].values
labels = df['label'].values

# 欧几里得距离矩阵
dist_matrix = cdist(coords, coords, metric='euclidean')

# 平均距离（上三角去除重复）
upper_tri = dist_matrix[np.triu_indices(len(coords), k=1)]
mean_distance = np.mean(upper_tri)

print(f"✅ 所有点之间的平均距离: {mean_distance:.4f}")

# 获取唯一标签
unique_labels = df['label'].unique()

# 每类标签之间的最小值与平均最小距离
print("\n✅ 各类标签之间的最小距离与平均最小距离：")
for i, label1 in enumerate(unique_labels):
    coords1 = df[df['label'] == label1][['x', 'y']].values
    for j, label2 in enumerate(unique_labels):
        if i >= j:
            continue  # 避免重复和自身
        coords2 = df[df['label'] == label2][['x', 'y']].values

        distances = cdist(coords1, coords2)

        min_dist = np.min(distances)
        mean_min_dist = np.mean(np.min(distances, axis=1))  # 每个点到另一类最近点的距离，然后平均

        print(f"{label1} ↔ {label2}: 最小距离 = {min_dist:.4f}, 平均最小距离 = {mean_min_dist:.4f}")
