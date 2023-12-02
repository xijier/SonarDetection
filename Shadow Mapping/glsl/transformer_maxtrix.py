
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

mesh = o3d.io.read_triangle_mesh("08.off")  # 读取.off文件
mesh.compute_vertex_normals()  # 计算mesh的法向量
mesh.paint_uniform_color([0.8, 0.8, 0.8])  # 上色，方便可视化
mesh.scale(4.0, (0.0, 0.0, 0.0))
mesh.translate([0, 0, 0.9])
R = mesh.get_rotation_matrix_from_xyz((0.7 * np.pi, 0, 0.6 * np.pi))
#R = mesh.get_rotation_matrix_from_xyz((0, 0.5, 0))


transform = mesh.get_rotation_matrix_from_xyz((0, 0.5, 0))
print(R)
b = np.zeros(3)
tran = np.insert(R, 3, values=b, axis=1)
print(tran)
c = [0,0,0,1]
tran1 = np.insert(tran, 3, values=c, axis=0)
# tran1 = np.r_[tran, np.zeros(3)]
print(tran1)


