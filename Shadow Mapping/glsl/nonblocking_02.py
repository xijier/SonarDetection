import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

class MySceneWidget:

    def __init__(self):

        self.w = gui.Application.instance.create_window("Open3D Example - Events", 800, 800)
        self.scene = gui.SceneWidget()
        self.scene.scene = o3d.visualization.rendering.Open3DScene(self.w.renderer)
        self.w.add_child(self.scene)
        #self.material = o3d.visualization.rendering.MaterialRecord()

        self.bbox = o3d.geometry.AxisAlignedBoundingBox([-2, -2, -2], [2, 2, 2])

        self.scene.setup_camera(60, self.bbox, [0, 0, 0])

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"

        #
        #大地
        self.ground = o3d.geometry.TriangleMesh.create_box(width=5, height=5, depth=0.1).subdivide_midpoint(10)
        self.ground.compute_vertex_normals()
        self.ground.translate([-2, -2, 0])
        self.ground.paint_uniform_color([1, 1, 1])
        self.scene.scene.add_geometry("ground", self.ground, self.material)

        #光源
        self.sun = o3d.geometry.TriangleMesh.create_sphere(0.1,20,False)
        self.sun.compute_vertex_normals()
        self.sun.translate([-0.4, 1.4, 1])
        self.sun.paint_uniform_color([1, 1, 1])
        self.scene.scene.add_geometry("sun", self.sun, self.material)


        self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))
        self.scene.set_on_mouse(self.on_mouse)
        self.scene.set_on_key(self.on_key)
        self.scene.scene.scene.enable_indirect_light(False)
        self.scene.scene.scene.enable_sun_light(False)
        self.scene.set_on_sun_direction_changed(self.set_light_dir)
        self.set_light_dir(np.array([-0.4, 1.4, -1]))
        self.light_xyz = [0,0,0]
        self.l_x = 0
        self.l_y = 0
        self.l_z = 0

    def set_light_dir(self,light_dir):
        self.scene.scene.scene.remove_light('light')
        self.scene.scene.scene.add_directional_light('light', [1, 1, 1], light_dir, 1e5, True)

    def on_key(self,e):

        if e.key == gui.KeyName.W:
            self.l_y += 1.0
            aa = self.sun.get_rotation_matrix_from_xyz((0, 0, 0))
            print(aa)
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            transform = [[1,0,0,-0.4],
                         [0,1,0,1.4+self.l_y],
                         [0,0,1,1],
                         [0, 0 , 0,1]]
            T = np.eye(4)
            print(T)
            T[1, 3] = self.l_y
            print(T)
            self.scene.scene.set_geometry_transform("sun", transform)
            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.S:

            self.l_y -= 1.0
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            transform = [[1,0,0,-0.4],
                         [0,1,0,1.4+self.l_y],
                         [0,0,1,1],
                         [0, 0 , 0,1]]

            self.scene.scene.set_geometry_transform("sun", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.A:
            self.l_x+= 0.1
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.D:
            self.l_x -= 0.1
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.Z:
            self.l_z+= 0.1
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.Q:
            self.l_z -= 0.1
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.SPACE:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN
                self.x += 0.1
                R = self.mesh.get_rotation_matrix_from_xyz((self.x, 0, 0))
                print(R)
                b = np.zeros(3)
                tran = np.insert(R, 3, values=b, axis=1)
                print(tran)
                c = [0, 0, 0, 1]
                transform = np.insert(tran, 3, values=c, axis=0)
                # tran1 = np.r_[tran, np.zeros(3)]
                print(transform)
                self.scene.scene.set_geometry_transform("airplane",transform)
                print("[debug] SPACE released")
            else:
                print("[debug] SPACE pressed")
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    def on_mouse(self, e):
        if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
            print("[debug] mouse:", (e.x, e.y))
        return gui.Widget.EventCallbackResult.IGNORED

def main():
    # gui.Application.instance.initialize()
    gui.Application.instance.initialize()
    myscene = MySceneWidget()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()