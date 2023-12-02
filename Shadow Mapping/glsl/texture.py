import sys, time, math, os, random

from pyglet.gl import *
import pyglet

window = pyglet.window.Window()
keyboard = pyglet.window.key.KeyStateHandler()
window.push_handlers(keyboard)


def loadTexture(filename):

    img = pyglet.image.load("style_0.jpg")
    data = img.get_data("RGBA", img.width * 4)
    textureIDs = (pyglet.gl.GLuint * 1)()
    glGenTextures(1, textureIDs)
    textureID = textureIDs[0]
    print('generating texture', textureID, 'from', filename)
    glBindTexture(GL_TEXTURE_2D, textureID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.height, img.width,
                 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glBindTexture(GL_TEXTURE_2D, 0)
    return textureID


class TexturedSquare:
    def __init__(self, width, height, xpos, ypos, texturefile):
        self.xpos = xpos
        self.ypos = ypos
        self.angle = 0
        self.size = 1
        self.texture = loadTexture(texturefile)
        x = width / 2.0
        y = height / 2.0
        #self.vlist = pyglet.graphics.vertex_list(4, ('v2f', [-x, -y, x, -y, -x, y, x, y]),
        #                                         ('t2f', [0, 0, 1, 0, 0, 1, 1, 1]))

    def draw(self):
        glPushMatrix()
        glTranslatef(self.xpos, self.ypos, 0)
        glRotatef(self.angle, 0, 0, 1)
        glScalef(self.size, self.size, self.size)
        glColor3f(1, 1, 1)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        #glColor4f(0.5, 0.5, 0.5, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(-30,30, -30)
        glVertex3f(-30, 30, 30)
        glVertex3f(30, 30, 30)
        glVertex3f(30, 30, -30)
        glEnd()
        #self.vlist.draw(GL_TRIANGLE_STRIP)
        sphere = gluNewQuadric()
        gluSphere(sphere, 100, 100, 100)
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()


@window.event
def on_draw():
    glClearColor(0, 0.3, 0.5, 0)
    glClear(GL_COLOR_BUFFER_BIT)
    square1.draw()


def update(dummy):
    global square1
    if keyboard[pyglet.window.key.A]:
        square1.xpos -= 5
    if keyboard[pyglet.window.key.D]:
        square1.xpos += 5
    if keyboard[pyglet.window.key.W]:
        square1.ypos += 5
    if keyboard[pyglet.window.key.S]:
        square1.ypos -= 5
    if keyboard[pyglet.window.key.UP]:
        square1.size *= 1.1
    if keyboard[pyglet.window.key.DOWN]:
        square1.size /= 1.1
    if keyboard[pyglet.window.key.LEFT]:
        square1.angle += 5
    if keyboard[pyglet.window.key.RIGHT]:
        square1.angle -= 5


square1 = TexturedSquare(120, 120, 300, 200, "style_0.jpg")

pyglet.clock.schedule_interval(update, 1 / 60.0)
pyglet.app.run()