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
pyglet.resource.path = ['data20']
pyglet.resource.reindex()


try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True,)
    #window = pyglet.window.Window(resizable=True, config=config, vsync=False) # "vsync=False" to check framerate
    window = pyglet.window.Window(width= 1280,height=960,resizable=True, config=config, vsync=False)
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)

render_width  = 640.0*2
render_height = 480.0*2
shadow_map_ratio = 2
# object angle
angle = 0
anglex = 0
angley = 0
anglez = 0

# Camera position
p_camera = [0,40,1]

# Camera lookAt
l_camera = [0,0,0]

# Light position
p_light = [-2,10,20]

# Light lookAt
l_light = [0,0,0]

# Light mouvement circle radius
light_mvnt = 20.0

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

def setupMatrices(position, lookAt):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, render_width/render_height, 10, 40000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(position[0], position[1], position[2],
              lookAt[0], lookAt[1], lookAt[2], 0, 1, 0)

@window.event
def on_resize(width, height):
    # Override the default on_resize handler to create a 3D projection
    if height==0: height=1
    glViewport(0, 0, width, height)
    setupMatrices(p_camera, l_camera)
    return pyglet.event.EVENT_HANDLED

autorot = 1
rottime = 0.0

def update(dt):
    global p_light
    global light_mvnt
    global rottime

    if autorot == 2:
        rottime += 0.5 * dt
    else:
        rottime += autorot * dt
    rottime = rottime % 360
    p_light[0] = light_mvnt * cos(rottime)
    p_light[2] = light_mvnt * sin(rottime)
    print(p_light)
pyglet.clock.schedule(update)

def setTextureMatrix():
    modelView = (GLdouble * 16)()
    projection = (GLdouble * 16)()
    bias = (GLdouble * 16)(0.5, 0.0, 0.0, 0.0,
                           0.0, 0.5, 0.0, 0.0,
                           0.0, 0.0, 0.5, 0.0,
                           0.5, 0.5, 0.5, 1.0)
# Grab modelview and transformation matrices
    glGetDoublev(GL_MODELVIEW_MATRIX, modelView)
    glGetDoublev(GL_PROJECTION_MATRIX, projection)
    glMatrixMode(GL_TEXTURE)
    glActiveTexture(GL_TEXTURE7)
    glLoadIdentity()
    glLoadMatrixd(bias)

# concatating all matrice into one.

    glMultMatrixd(projection)
    glMultMatrixd(modelView)

# Go back to normal matrix mode
    glMatrixMode(GL_MODELVIEW)

# During translation, we also have to maintain the GL_TEXTURE8, used in the shadow shader
# to determine if a vertex is in the shadow.

def startTranslate(x, y, z):
    glPushMatrix()
    glTranslatef(x,y,z)
    glMatrixMode(GL_TEXTURE)
    glActiveTexture(GL_TEXTURE7)
    glPushMatrix()
    glTranslatef(x,y,z)

def endTranslate():
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

def drawCoordinate():
  '''
  绘制三维的坐标系，并绘制由坐标轴构成的平面的网格，各个坐标轴的颜色以及由坐标轴所引出的网格线的颜色为：
  x: (1.0, 0.0, 0.0)
  y: (0.0, 1.0, 0.0)
  z: (0.0, 0.0, 1.0)
  :return:
  '''
  #设置网格线间的步长
  step = 0.2
  #设置网格线的数量、长度
  line_num = 15
  line_len = 6
  grid_color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

  glLineWidth(3)
  origin = [0.0, 0.0, 0.0]
  #画出xyz轴
  for i in range(3):
    tmp = [0.0, 0.0, 0.0]
    tmp[i] = line_len
    glColor3f(*grid_color[i])
    glBegin(GL_LINES)
    glVertex3f(*origin)
    glVertex3f(*tmp)
    glEnd()
  glLineWidth(1)

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
            #e_count = int(m[i][2])
            continue
        # 把字符型数据转化为数值型数据
        else:
            for j in range(len(m[i])):
                m[i][j] = float(m[i][j])
            n.append(m[i])
    return v_cout, f_count, n

def drawObjectsTexture(v_cout, f_count, n):
# Ground
    glColor4f(1, 1, 1, 1.0)
    #glColor4f(0.7, 0.7, 0.7, 1.0)
    glBegin(GL_QUADS)
    glVertex3f(-30, -1, -30)
    glVertex3f(-30, -1,  30)
    glVertex3f( 30, -1,  30)
    glVertex3f( 30, -1, -30)
    glEnd()
    glColor4f(0.9, 0.9, 0.9, 1.0)

    # startTranslate(0,-2.0,0)
    # drawCoordinate()
    # endTranslate()

    glPushMatrix()
    glColor4f(0.9, 0.9, 0.9, 1.0)

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, textureID)

    glRotatef(270, 1, 0, 0)
    #startTranslate(0, 0, 0)

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
        glVertex3f(a1*scaleratio, a2*scaleratio, a3*scaleratio)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(b1*scaleratio, b2*scaleratio, b3*scaleratio)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(c1*scaleratio, c2*scaleratio, c3*scaleratio)
        glEnd();
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()
    startTranslate(*p_light)
    sphere = gluNewQuadric()
    gluSphere(sphere, 0.5, 100, 100)
    endTranslate()

def drawObjects(v_cout, f_count, n):
# Ground
    glColor4f(0.7, 0.7, 0.7, 1.0)
    glBegin(GL_QUADS)
    glVertex3f(-30, -1, -30)
    glVertex3f(-30, -1,  30)
    glVertex3f( 30, -1,  30)
    glVertex3f( 30, -1, -30)
    glEnd()
    glColor4f(0.9, 0.9, 0.9, 1.0)
    # startTranslate(0,-2.0,0)
    # drawCoordinate()
    # endTranslate()
    glPushMatrix()
    glColor4f(0.9, 0.9, 0.9, 1.0)
    glRotatef(270, 1, 0, 0)
    #startTranslate(0, 0, 0)

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
        glVertex3f(a1*scaleratio, a2*scaleratio, a3*scaleratio)
        glVertex3f(b1*scaleratio, b2*scaleratio, b3*scaleratio)
        glVertex3f(c1*scaleratio, c2*scaleratio, c3*scaleratio)
        glEnd();
    glPopMatrix()
    startTranslate(*p_light)
    sphere = gluNewQuadric()
    gluSphere(sphere, 0.5, 100, 100)
    endTranslate()


@window.event
def on_draw():
# First step: Render from the light POV to a FBO, story depth values only
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fboId)  # Rendering offscreen

# In the case we render the shadowmap to a higher resolution, the viewport must be modified accordingly.
    glViewport( 0, 0, int(render_width*shadow_map_ratio), int(render_height*shadow_map_ratio) )

# Clear previous frame values
    glClear(GL_DEPTH_BUFFER_BIT)

# Disable color rendering, we only want to write to the Z-Buffer
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE)

    setupMatrices(p_light, l_light)

# Culling switching, rendering only backface, this is done to avoid self-shadowing
    glCullFace(GL_FRONT)


    #drawObjects(v_cout, f_count, n)
    drawObjectsTexture(v_cout, f_count, n)

# Save modelview/projection matrice into texture7, also add a biais
    setTextureMatrix()

# Now rendering from the camera POV, using the FBO to generate shadows
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0)
    glViewport(0, 0, int(render_width), int(render_height))

# Enabling color write (previously disabled for light POV z-buffer rendering)
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

# Clear previous frame values
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

# Using the shadow shader
    shader.bind()
    shader.uniformi('ShadowMap', 7)
    glActiveTexture(GL_TEXTURE7)
    glBindTexture(GL_TEXTURE_2D, depthTextureId)

    setupMatrices(p_camera, l_camera)
    glCullFace(GL_BACK)
    drawObjectsTexture(v_cout, f_count, n)
    shader.unbind()

@window.event
def on_key_press(symbol, modifiers):
    global autorot
    global p_camera
    global angle
    global anglez
    global anglex
    global angley

    if symbol == key.ESCAPE or symbol == key.Q:
        print ('Good Bye !')   # ESC would do it anyway, but not "Q"
        pyglet.app.exit()
        return pyglet.event.EVENT_HANDLED
    if symbol == key.SPACE:
        print ('Toggle Auto')
        autorot = (autorot + 1) % 3
    if symbol == key.A:
        print ('Camera left')
        p_camera[0] -= 2
    if symbol == key.D:
        print ('Camera right')
        p_camera[0] += 2
    if symbol == key.W:
        print ('Camera up')
        p_camera[1] += 2
    if symbol == key.S:
        print ('Camera down')
        p_camera[1] -= 2
    if symbol == key.R:
        print ('Camera reset')
        p_camera = [32,20,0]
    if symbol == key.UP:
        print ('light up :'+str(p_light))
        p_light[1] += 1
    if symbol == key.DOWN:
        print ('light down:' +str(p_light))
        p_light[1] -= 1
    if symbol == key.Z:
        print ('anglez')
        anglex = 0
        angley = 0
        anglez = 1
    if symbol == key.X:
        print ('angleX')
        anglex = 1
        angley = 0
        anglez = 0
    if symbol == key.Y:
        print ('angleX')
        anglex = 0
        angley = 1
        anglez = 0

    if symbol == key.T:
        print('angle: '+ str(angle))
        angle += 20

shader = Shader(['''
varying vec4 ShadowCoord;

void main()
  {
  ShadowCoord = gl_TextureMatrix[7] * gl_Vertex;
  gl_Position = ftransform();
  gl_FrontColor = gl_Color;
  }
'''], ['''
varying vec4 ShadowCoord;
uniform sampler2D ShadowMap;

void main()
  {
  vec4 shadowCoordinateWdivide = ShadowCoord / ShadowCoord.w ;
  vec3 color = gl_Color.xyz;

//Used to lower moire pattern and self-shadowing
//shadowCoordinateWdivide.z += 0.0005;
  float distanceFromLight = texture2D(ShadowMap,shadowCoordinateWdivide.xy).z;
  float shadow = 1.0; // no shadow
  if (ShadowCoord.w > 0.0)
    {
//  if ( distanceFromLight < shadowCoordinateWdivide.z )
//    shadow = 0.5;       // in shadow
    if ( shadowCoordinateWdivide.z > 0.0001 )
      shadow = 0.5 + 0.5 * smoothstep( 0.997 , 1.003, distanceFromLight / shadowCoordinateWdivide.z );
    }
  gl_FragColor = shadow * vec4(color, 1.0);
  }
'''])

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

generateShadowFBO()
# This is important, if not here, FBO's depthbuffer won't be populated.
glEnable(GL_DEPTH_TEST)
glClearColor(0.0,0.0,0.0,1.0)
glEnable(GL_CULL_FACE)
glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
textureID = loadTexture("style_1.jpg")
v_cout, f_count, n = readOFF('01.off')
pyglet.app.run()