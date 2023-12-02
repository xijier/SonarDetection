import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import cv2

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

        #大地
        self.ground = o3d.geometry.TriangleMesh.create_box(width=7, height=7, depth=0.1).subdivide_midpoint(10)
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

        # 飞机
        #self.mesh = o3d.io.read_triangle_mesh("07.off")  # 读取.off文件
        self.mesh = o3d.io.read_triangle_mesh("./data/ariplane/01.off")  # 读取.off文件
        self.mesh.compute_vertex_normals()  # 计算mesh的法向量
        self.mesh.paint_uniform_color([0.9, 0.9, 0.9])  # 上色，方便可视化
        #self.mesh.scale(0.12, (0.0, 0.0, 0.0))
        self.mesh.scale(4, (0.0, 0.0, 0.0))
        #self.mesh.translate([0, 0, 0.9])
        self.mesh.translate([0, 0, 0.4])
        #R = self.mesh.get_rotation_matrix_from_xyz((0.7 * np.pi, 0, 0.6 * np.pi))
        self.R = self.mesh.get_rotation_matrix_from_xyz((0, 0, 0))
        self.mesh.rotate(self.R, center=(0, 0, 0))
        self.scene.scene.add_geometry("airplane", self.mesh, self.material)

        self.scene.setup_camera(60, self.scene.scene.bounding_box, (0, 0, 0))
        self.scene.set_on_mouse(self._on_mouse_widget3d)
        self.scene.set_on_key(self.on_key)
        self.scene.scene.scene.enable_indirect_light(False)
        self.scene.scene.scene.enable_sun_light(True)
        # self.scene.scene.scene.enable_light_shadow("airplane",False)
        self.scene.set_on_sun_direction_changed(self.set_light_dir)

        self.text = cv2.imread('./data/QwIKM.png')
        self.mesh.textures = [o3d.geometry.Image(self.text)]

        DX, DY = 0.5 / 2, 0.66 / 2
        v_uv = [[DX, DY], [DX, 2 * DY], [2 * DX, 2 * DY], [2 * DX, DY],
                [0, DX], [DX, 1], [3 * DX, 2 * DY], [3 * DX, DY]]

        v_uv = np.asarray(v_uv)
        v_uv = np.concatenate((v_uv, v_uv, v_uv), axis=0)
        #self.mesh.triangle_uvs = o3d.open3d_pybind.utility.Vector2dVector(v_uv)

        self.mesh.textures = [o3d.geometry.Image(self.text)]

        self.set_light_dir(np.array([-0.1, 1.4, -0.4]))
        print([-0.4, 1.4, -1])
        self.l_x = 0
        self.l_y = 0
        self.l_z = 0
        self.o_x = 0
        self.o_y = 0
        self.o_z = 0
        self.o_x_t = 0
        self.o_y_t = 0
        self.o_z_t = 0

    def set_light_dir(self,light_dir):
        self.scene.scene.scene.remove_light('light')
        self.scene.scene.scene.add_directional_light('light', [1, 1, 1], light_dir, 1e5, True)

    def on_key(self,e):

        # obj control

        if e.key == gui.KeyName.UP:
            self.o_x += 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            R = self.mesh.get_rotation_matrix_from_xyz((self.o_x, self.o_y, self.o_z))
            #print(R)
            b = np.zeros(3)
            tran = np.insert(R, 3, values=b, axis=1)
            #print(tran)
            c = [0, 0, 0, 1]
            transform = np.insert(tran, 3, values=c, axis=0)
            # tran1 = np.r_[tran, np.zeros(3)]
            #print(transform)
            self.scene.scene.set_geometry_transform("airplane", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.DOWN:
            self.o_x -= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            R = self.mesh.get_rotation_matrix_from_xyz((self.o_x, self.o_y, self.o_z))
            print(R)
            b = np.zeros(3)
            tran = np.insert(R, 3, values=b, axis=1)
            c = [0, 0, 0, 1]
            transform = np.insert(tran, 3, values=c, axis=0)
            # tran1 = np.r_[tran, np.zeros(3)]
            print(transform)
            self.scene.scene.set_geometry_transform("airplane", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.RIGHT:
            self.o_y += 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            R = self.mesh.get_rotation_matrix_from_xyz((self.o_x, self.o_y, self.o_z))
            print(R)
            b = np.zeros(3)
            tran = np.insert(R, 3, values=b, axis=1)
            print(tran)
            c = [0, 0, 0, 1]
            transform = np.insert(tran, 3, values=c, axis=0)
            # tran1 = np.r_[tran, np.zeros(3)]
            print(transform)
            self.scene.scene.set_geometry_transform("airplane", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.LEFT:
            self.o_y -= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            R = self.mesh.get_rotation_matrix_from_xyz((self.o_x, self.o_y, self.o_z))
            print(R)
            b = np.zeros(3)
            tran = np.insert(R, 3, values=b, axis=1)
            print(tran)
            c = [0, 0, 0, 1]
            transform = np.insert(tran, 3, values=c, axis=0)
            # tran1 = np.r_[tran, np.zeros(3)]
            print(transform)
            self.scene.scene.set_geometry_transform("airplane", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.N:
            self.o_z += 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            R = self.mesh.get_rotation_matrix_from_xyz((self.o_x, self.o_y, self.o_z))
            print(R)
            b = np.zeros(3)
            tran = np.insert(R, 3, values=b, axis=1)
            print(tran)
            c = [0, 0, 0, 1]
            transform = np.insert(tran, 3, values=c, axis=0)
            # tran1 = np.r_[tran, np.zeros(3)]
            print(transform)
            self.scene.scene.set_geometry_transform("airplane", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.M:
            self.o_z -= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            R = self.mesh.get_rotation_matrix_from_xyz((self.o_x, self.o_y, self.o_z))
            print(R)
            b = np.zeros(3)
            tran = np.insert(R, 3, values=b, axis=1)
            print(tran)
            c = [0, 0, 0, 1]
            transform = np.insert(tran, 3, values=c, axis=0)
            # tran1 = np.r_[tran, np.zeros(3)]
            print(transform)
            self.scene.scene.set_geometry_transform("airplane", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        #light control
        if e.key == gui.KeyName.W:
            self.l_y += 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            transform = [[1, 0, 0, -0.4+ self.l_x],
                         [0, 1, 0, 1.4 + self.l_y],
                         [0, 0, 1, 1+ self.l_z],
                         [0, 0, 0, 1]]
            self.scene.scene.set_geometry_transform("sun", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.S:
            self.l_y -= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))
            transform = [[1, 0, 0, -0.4+ self.l_x],
                         [0, 1, 0, 1.4 + self.l_y],
                         [0, 0, 1, 1+ self.l_z],
                         [0, 0, 0, 1]]
            self.scene.scene.set_geometry_transform("sun", transform)
            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.A:
            self.l_x+= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))

            transform = [[1, 0, 0, -0.4+ self.l_x],
                         [0, 1, 0, 1.4 + self.l_y],
                         [0, 0, 1, 1+ self.l_z],
                         [0, 0, 0, 1]]
            self.scene.scene.set_geometry_transform("sun", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.D:
            self.l_x -= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))

            transform = [[1, 0, 0, -0.4+ self.l_x],
                         [0, 1, 0, 1.4 + self.l_y],
                         [0, 0, 1, 1+ self.l_z],
                         [0, 0, 0, 1]]
            self.scene.scene.set_geometry_transform("sun", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.Z:
            self.l_z+= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))

            transform = [[1, 0, 0, -0.4+ self.l_x],
                         [0, 1, 0, 1.4 + self.l_y],
                         [0, 0, 1, 1+ self.l_z],
                         [0, 0, 0, 1]]
            self.scene.scene.set_geometry_transform("sun", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED
        if e.key == gui.KeyName.Q:
            self.l_z -= 0.1
            print("light: " + str([-0.4 + self.l_x, 1.4 + self.l_y, -1 + self.l_z]))
            self.set_light_dir(np.array([-0.4 + self.l_x, 1.4+ self.l_y, -1+self.l_z]))

            transform = [[1, 0, 0, -0.4+ self.l_x],
                         [0, 1, 0, 1.4 + self.l_y],
                         [0, 0, 1, 1+ self.l_z],
                         [0, 0, 0, 1]]
            self.scene.scene.set_geometry_transform("sun", transform)

            print("[debug] Eating W")
            return gui.Widget.EventCallbackResult.CONSUMED

        if e.key == gui.KeyName.SPACE:
            if e.type == gui.KeyEvent.UP:  # check UP so we default to DOWN

                self.scene.scene.remove_geometry("sun")
                # self.x += 0.1
                # R = self.mesh.get_rotation_matrix_from_xyz((self.x, 0, 0))
                # print(R)
                # b = np.zeros(3)
                # tran = np.insert(R, 3, values=b, axis=1)
                # print(tran)
                # c = [0, 0, 0, 1]
                # transform = np.insert(tran, 3, values=c, axis=0)
                # # tran1 = np.r_[tran, np.zeros(3)]
                # print(transform)
                # self.scene.scene.set_geometry_transform("airplane",transform)
                print("[debug] SPACE released")
            else:
                print("[debug] SPACE pressed")
            return gui.Widget.EventCallbackResult.HANDLED

    def _on_mouse_widget3d(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self.scene.frame.x
                y = event.y - self.scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                else:
                    world = self.scene.scene.camera.unproject(
                        event.x, event.y, depth, self.scene.frame.width,
                        self.scene.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                # def update_label():
                #     self.info.text = text
                #     self.info.visible = (text != "")
                #     # We are sizing the info label to be exactly the right size,
                #     # so since the text likely changed width, we need to
                #     # re-layout to set the new frame.
                #     self.window.set_needs_layout()
                #
                # gui.Application.instance.post_to_main_thread(
                #     self.window, update_label)

            self.scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    # def on_mouse(self, e):
    #     # if e.type == gui.MouseEvent.Type.MOVE:
    #     #     print("[debug] mouse:", (e.x, e.y))
    #
    #     if e.type == gui.MouseEvent.Type.BUTTON_DOWN:
    #         print("[debug] mouse:", (e.x, e.y))
    #
    #     if e.is_button_down(gui.MouseButton.LEFT):
    #
    #         def depth_callback(depth_image):
    #             # Coordinates are expressed in absolute coordinates of the
    #             # window, but to dereference the image correctly we need them
    #             # relative to the origin of the widget. Note that even if the
    #             # scene widget is the only thing in the window, if a menubar
    #             # exists it also takes up space in the window (except on macOS).
    #             x = e.x - self.scene.frame.x
    #             y = e.y - self.scene.frame.y
    #             # Note that np.asarray() reverses the axes.
    #             depth = np.asarray(depth_image)[y, x]
    #
    #             if depth == 1.0:  # clicked on nothing (i.e. the far plane)
    #                 text = ""
    #             else:
    #                 world = self.scene.scene.camera.unproject(
    #                     e.x, e.y, depth, self.scene.frame.width,
    #                     self.scene.frame.height)
    #                 text = "({:.3f}, {:.3f}, {:.3f})".format(
    #                     world[0], world[1], world[2])
    #
    #                 print(text)
    #         self.scene.scene.scene.render_to_depth_image(depth_callback)
    #         return gui.Widget.EventCallbackResult.HANDLED
    #
    #     return gui.Widget.EventCallbackResult.IGNORED


def main():
    # gui.Application.instance.initialize()
    gui.Application.instance.initialize()
    myscene = MySceneWidget()
    gui.Application.instance.run()

if __name__ == "__main__":
    main()