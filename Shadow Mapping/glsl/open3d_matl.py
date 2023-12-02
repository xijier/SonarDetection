import open3d as o3d
import copy
import numpy as np
# mesh = o3d.io.read_triangle_mesh("./data/body4.off") #读取.off文件
# mesh.compute_vertex_normals()  #计算mesh的法向量
# mesh.paint_uniform_color([0.7, 0.7, 0.7])  #上色，方便可视化
# o3d.visualization.draw_geometries([mesh])

# import open3d as o3d
# import numpy as np
# pcd = o3d.io.read_point_cloud("./data/body4.txt")
# print(pcd)#输出点云点的个数
# print(np.asarray(pcd.points))#输出点的三维坐标
# print('给所有的点上一个统一的颜色，颜色是在RGB空间得[0，1]范围内得值')
# pcd.paint_uniform_color([0, 1, 0])
# o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
# o3d.visualization.draw_geometries([pcd])

import open3d as o3d
import numpy as np
pcd = o3d.io.read_point_cloud("./data/test.txt",format="xyz")
print(pcd)#输出点云点的个数
print(np.asarray(pcd.points))#输出点的三维坐标
print('给所有的点上一个统一的颜色，颜色是在RGB空间得[0，1]范围内得值')
#pcd.paint_uniform_color([0, 1, 0])
#o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
o3d.visualization.draw_geometries([pcd])


