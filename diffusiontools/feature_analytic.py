import pandas as pd

# Load the uploaded t-SNE data
file_path = "output_umap_ship.xlsx"
df = pd.read_excel(file_path)

df.head(), df["label"].value_counts()

import numpy as np

# Compute centroids
# centroids = df.groupby("label")[["tsne1", "tsne2"]].mean()
#
# # Compute intra-class dispersion (mean distance to centroid)
# def mean_distance_to_centroid(group):
#     c = group[["tsne1","tsne2"]].mean().values
#     pts = group[["tsne1","tsne2"]].values
#     return np.mean(np.linalg.norm(pts - c, axis=1))


# Compute centroids
centroids = df.groupby("label")[["umap1", "umap2"]].mean()

# Compute intra-class dispersion (mean distance to centroid)
def mean_distance_to_centroid(group):
    c = group[["umap1","umap2"]].mean().values
    pts = group[["umap1","umap2"]].values
    return np.mean(np.linalg.norm(pts - c, axis=1))


dispersion = df.groupby("label").apply(mean_distance_to_centroid)

# Compute inter-class centroid distances
labels = centroids.index.tolist()
dist_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

for i in labels:
    for j in labels:
        dist_matrix.loc[i, j] = np.linalg.norm(
            centroids.loc[i].values - centroids.loc[j].values
        )

print(centroids)
print(dispersion)
print(dist_matrix)
