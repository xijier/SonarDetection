import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import numpy as np

use_point_light = False

gui.Application.instance.initialize()
window = gui.Application.instance.create_window("use_point_light", 500, 500)
scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(window.renderer)
window.add_child(scene_widget)
bbox = o3d.geometry.AxisAlignedBoundingBox([-2, -2, -2], [2, 2, 2])
scene_widget.setup_camera(60, bbox, [0, 0, 0])


material = rendering.MaterialRecord()
# material.base_color = [.5,1,.5,1]
material.shader = "defaultLit"

box1 = o3d.geometry.TriangleMesh.create_box(width=5,height=5,depth=.1).subdivide_midpoint(10)
box1.compute_vertex_normals()
box1.paint_uniform_color([0.1, 0.1, 0.7])
scene_widget.scene.add_geometry("box1", box1, material)

# box2 = o3d.geometry.TriangleMesh.create_box(width=.1,height=.1,depth=.3).translate([0,0,.1]).subdivide_midpoint(3)
# box2.compute_vertex_normals()
# box2.paint_uniform_color([0.1, 0.1, 0.7])
# scene_widget.scene.add_geometry("box2", box2, material)

mesh = o3d.io.read_triangle_mesh("08.off") #读取.off文件
mesh.compute_vertex_normals()  #计算mesh的法向量
mesh.paint_uniform_color([0.7, 0.7, 0.7])  #上色，方便可视化
mesh.scale(4.0, (0.0, 0.0, 0.0))
mesh.translate([0, 0, 1])
#o3d.visualization.draw_geometries([mesh])
scene_widget.scene.add_geometry("ss",mesh,material)


scene_widget.scene.scene.enable_indirect_light(False)
scene_widget.scene.scene.enable_sun_light(False)

def set_light_dir(light_dir):
    #scene_widget.scene.scene.remove_light('light')
    if use_point_light:
        print("Adding spot light")
        # scene_widget.scene.scene.add_spot_light('light',[1,1,1],[-1, -1, 1.5],light_dir,1e6,1e2,.1,.2,True)
        scene_widget.scene.scene.add_point_light('light',[1,1,1], [-0.5, -0.5, 1.0],1e6,1e2,True)
    else:
        scene_widget.scene.scene.add_directional_light('light',[1,1,1],light_dir,1e5,True)


scene_widget.set_on_sun_direction_changed(set_light_dir)

set_light_dir(np.array([1,1,-1]))
gui.Application.instance.run()