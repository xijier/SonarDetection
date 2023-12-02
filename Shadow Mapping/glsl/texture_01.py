#!/usr/bin/env python
''' THIS IS WORK IN PROGRESS - USE WITH CARE

Displays a simple FBO shadow map
    adapted by Pythonese 2009 for Pyglet line by line from
        http://www.fabiensanglard.net/shadowmapping/index.php

    for FBO we need:
        glGenFramebuffersEXT
        glBindFramebufferEXT
        glFramebufferTexture2DEXT
        glCheckFramebufferStatusEXT
'''

from math import pi, sin, cos, sqrt
from euclid import *
import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet import image, resource
import os
from shader import Shader
from ctypes import *

try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True,)
    window = pyglet.window.Window(resizable=True, config=config, vsync=False) # "vsync=False" to check framerate
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)

render_width = 640.0
render_height = 480.0
shadow_map_ratio = 2

# Camera position
p_camera = [0, 20, 2]
# Camera lookAt
l_camera = [0, 0, 0]

# Hold id of the framebuffer for light POV rendering
fboId = GLuint()
# Z values will be rendered to this texture when using fboId framebuffer
depthTextureId = GLuint()

def generateShadowFBO():
    global depthTextureId
    global fboId

    shadowMapWidth  = int(render_width * shadow_map_ratio)
    shadowMapHeight = int(render_height * shadow_map_ratio)

    glGenTextures(1, byref(depthTextureId))
    depthTextureId = depthTextureId.value
    glBindTexture(GL_TEXTURE_2D, depthTextureId)

# GL_LINEAR does not make sense for depth texture. However, next tutorial ?
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

# Remove artefact on the edges of the shadowmap
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)

# No need to force GL_DEPTH_COMPONENT24, drivers usually give you the max precision if available

    glTexImage2D( GL_TEXTURE_2D, 0,
                  GL_DEPTH_COMPONENT, # texture format
                  shadowMapWidth,
                  shadowMapHeight,
                  0,
                  GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE,
                  0)
    glBindTexture(GL_TEXTURE_2D, 0)

# create a framebuffer object
    glGenFramebuffersEXT(1, byref(fboId))
    fboId = fboId.value
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId)

# Instruct openGL that we won't bind a color texture with the currently binded FBO
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)

# attach the texture to FBO depth attachment point
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,GL_TEXTURE_2D, depthTextureId, 0)

# check FBO status
    FBOstatus = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)
    if FBOstatus != GL_FRAMEBUFFER_COMPLETE_EXT:
        print ("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use FBO")

# switch back to window-system-provided framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0)

def readOFF(path):
    file_path = path
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    m = []
    n = []
    # 把off文件中数据按行读取出来，存储到列表中,由于边数事先看了为0，所以没遍历边，如果边不为0，记得遍历，并在后文加上边的绘制
    for line in lines:
        m.append(line.split())
    for i in range(len(m)):
        # 跳过第一行的OFF
        if m[i][0] == 'OFF':
            continue
        # 记录定点数，面片数和边数
        elif i == 1:
            v_cout = int(m[i][0])
            f_count = int(m[i][1])
            # e_count = int(m[i][2])
            continue
        # 把字符型数据转化为数值型数据
        else:
            for j in range(len(m[i])):
                m[i][j] = float(m[i][j])
            n.append(m[i])
    return v_cout, f_count, n

def drawObjects(v_cout, f_count, n):

    glColor3f(0.6, 0.6, 0.6)
    glBegin(GL_QUADS)
    # 设置纹理坐标
    # 绘制顶点
    glVertex3f(-100.0, -100.0, 1.0)
    glVertex3f(100.0, -100.0, 1.0)
    glVertex3f(100.0, 100.0, 1.0)
    glVertex3f(-100.0, 100.0, 1.0)
    glEnd()

    glPushMatrix()
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textureID)
    #glRotatef(90, 1, 0, 0)
    for i in range(f_count):
        # 获取顶点个数，和顶点信息
        cout, a, b, c = n[v_cout + i]
        cout, a, b, c = int(cout), int(a), int(b), int(c)
        # 得到顶点位置
        a1, a2, a3 = n[a]
        b1, b2, b3 = n[b]
        c1, c2, c3 = n[c]
        # 绘制多面体
        scaleratio = 15
        glBegin(GL_POLYGON)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(a1 * scaleratio, a2 * scaleratio, a3 * scaleratio)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(b1 * scaleratio, b2 * scaleratio, b3 * scaleratio)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(c1 * scaleratio, c2 * scaleratio, c3 * scaleratio)
        glEnd();
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()

def loadTexture(filename):

    img = pyglet.image.load(filename)
    data = img.get_data("RGB", img.width * 3)
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

def setupMatrices(position, lookAt):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, render_width/render_height, 10, 40000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(position[0], position[1], position[2],
              lookAt[0], lookAt[1], lookAt[2], 0, 1, 0)

@window.event
def on_draw():
    glClearColor(0, 0.3, 0.5, 0)
    glViewport(0, 0, int(render_width * shadow_map_ratio), int(render_height * shadow_map_ratio))
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    setupMatrices(p_camera, l_camera)

    v_cout, f_count, n = readOFF('01.off')
    drawObjects(v_cout, f_count, n)

textureID = loadTexture("style_0.jpg")
pyglet.app.run()