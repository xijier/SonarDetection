# -*- coding:utf-8 -*-
import numpy as np
import open3d as o3d

# 设置Open3D调试级别
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
# 加载点云
source_raw = o3d.io.read_point_cloud("./DemoICPPointClouds/cloud_bin_1.pcd")
target_raw = o3d.io.read_point_cloud("./DemoICPPointClouds/cloud_bin_2.pcd")
# 体素下采样
source = source_raw.voxel_down_sample(voxel_size=0.02)
target = target_raw.voxel_down_sample(voxel_size=0.02)
# 原始点云空间转换
trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
source.transform(trans)
# 翻转
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
source.transform(flip_transform)
target.transform(flip_transform)
# 可视化
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(source)
vis.add_geometry(target)
threshold = 0.05
icp_iteration = 100
save_image = False

for i in range(icp_iteration):
    # 配准
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
    source.transform(reg_p2l.transformation)
    vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()
    if save_image:
        vis.capture_screen_image("temp_%04d.jpg" % i)
vis.destroy_window()