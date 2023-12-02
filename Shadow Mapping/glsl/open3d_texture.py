import os
import cv2
import numpy as np
import open3d as o3d

vert = [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
        [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]]

faces = [[3, 0, 1], [1, 2, 3],
         [1, 5, 6], [6, 2, 1],
         [5, 4, 7], [7, 6, 5],
         [4, 0, 3], [3, 7, 4],
         [6, 7, 3], [3, 2, 6],
         [1, 0, 4], [4, 5, 1]]

m = o3d.geometry.TriangleMesh(o3d.open3d.utility.Vector3dVector(vert),
                              o3d.open3d.utility.Vector3iVector(faces))

m.compute_vertex_normals()

text = cv2.imread('./data/QwIKM.png')

DX, DY = 0.5 / 2, 0.66 / 2  # UV坐标

one = [[3 * DX, 1 * DY], [3 * DX, 2 * DY], [4 * DX, 2 * DY], [4 * DX, 2 * DY], [4 * DX, 1 * DY], [3 * DX, 1 * DY]]
two = [[2 * DX, 1 * DY], [2 * DX, 2 * DY], [3 * DX, 2 * DY], [3 * DX, 2 * DY], [3 * DX, 1 * DY], [2 * DX, 1 * DY]]
three = [[1 * DX, 1 * DY], [1 * DX, 2 * DY], [2 * DX, 2 * DY], [2 * DX, 2 * DY], [2 * DX, 1 * DY], [1 * DX, 1 * DY]]
four = [[0 * DX, 1 * DY], [0 * DX, 2 * DY], [1 * DX, 2 * DY], [1 * DX, 2 * DY], [1 * DX, 1 * DY], [0 * DX, 1 * DY]]
five = [[1 * DX, 0 * DY], [1 * DX, 1 * DY], [2 * DX, 1 * DY], [2 * DX, 1 * DY], [2 * DX, 0 * DY], [1 * DX, 0 * DY]]
six = [[1 * DX, 2 * DY], [1 * DX, 3 * DY], [2 * DX, 3 * DY], [2 * DX, 3 * DY], [2 * DX, 2 * DY], [1 * DX, 2 * DY]]

v_uv = np.concatenate((one, two, three, four, five, six), axis=0)
print(v_uv.shape)  # (48, 2)

m.triangle_uvs = o3d.open3d.utility.Vector2dVector(v_uv)
m.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
m.textures = [o3d.geometry.Image(text)]

o3d.visualization.draw_geometries([m])