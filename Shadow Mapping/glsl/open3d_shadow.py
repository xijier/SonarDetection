import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import numpy as np

use_point_light = False

sun_pos = [1,1,-1]

gui.Application.instance.initialize()
window = gui.Application.instance.create_window("use_point_light", 800, 800)
scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
window.add_child(scene_widget)
bbox = o3d.geometry.AxisAlignedBoundingBox([-2, -2, -2], [2, 2, 2])

scene_widget.setup_camera(60, bbox, [0, 0, 0])


material = rendering.MaterialRecord()
# material.base_color = [.5,1,.5,1]
material.shader = "defaultLit"

# origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
# scene_widget.scene.add_geometry("coordinate", origin, material)

ground = o3d.geometry.TriangleMesh.create_box(width=5,height=5,depth=0.1).subdivide_midpoint(10)
ground.compute_vertex_normals()
ground.translate([-2, -2, 0])
ground.paint_uniform_color([1, 1, 1])
scene_widget.scene.add_geometry("ground", ground, material)

# sun = o3d.geometry.TriangleMesh.create_sphere(0.2,20,False)
# sun.compute_vertex_normals()
# sun.translate(sun_pos)
# sun.paint_uniform_color([1.0, 0.9, 0.9])
# scene_widget.scene.add_geometry("sun", sun, material)

mesh = o3d.io.read_triangle_mesh("08.off") #读取.off文件
mesh.compute_vertex_normals()  #计算mesh的法向量
mesh.paint_uniform_color([0.8, 0.8, 0.8])  #上色，方便可视化
mesh.scale(4.0, (0.0, 0.0, 0.0))
mesh.translate([0, 0, 0.9])
R = mesh.get_rotation_matrix_from_xyz((0.7 * np.pi, 0, 0.6 * np.pi))
R = mesh.get_rotation_matrix_from_xyz((0, 0.5, 0))
mesh.rotate(R, center=(0,0,0))
scene_widget.scene.add_geometry("airplane",mesh,material)

scene_widget.scene.scene.enable_indirect_light(False)
scene_widget.scene.scene.enable_sun_light(False)


def set_light_dir(light_dir):
    scene_widget.scene.scene.remove_light('light')
    scene_widget.scene.scene.add_directional_light('light',[1,1,1],light_dir,1e5,True)

scene_widget.set_on_sun_direction_changed(set_light_dir)
set_light_dir(np.array([-0.4,1.4,-1]))
gui.Application.instance.run()