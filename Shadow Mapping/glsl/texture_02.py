import sys, time, math, os, random

from pyglet.gl import *
import pyglet

window = pyglet.window.Window(width=800, height=600)
#keyboard = pyglet.window.key.KeyStateHandler()
#window.push_handlers(keyboard)

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

@window.event
def on_draw():
    glClearColor(0, 0.3, 0.5, 0)
    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800 / 600, 10, 40000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(200, 200, 500,
              200, 200, 0,
              0, 1, 0)

    img = pyglet.image.load('style_0.jpg')
    tex = img.get_texture()
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    v_cout, f_count, n = readOFF('01.off')
    # glColor4f(0.5, 0.5, 0.5, 1.0)
    # glBegin(GL_QUADS)
    # glVertex3f(-100, -100, 0)
    # glVertex3f(-100, 100, 0)
    # glVertex3f(100, 100, 0)
    # glVertex3f(100, -100, 0)
    # glEnd()

    glColor4f(0.9, 0.9, 0.9, 1.0)
    pyglet.gl.glEnable(tex.target)
    pyglet.gl.glBindTexture(tex.target, tex.id)
    glPushMatrix()
    glTranslatef(300, 300, 0)
    glRotatef(270, 1, 0, 0)
    for i in range(f_count):
        # 获取顶点个数，和顶点信息
        cout, a, b, c = n[v_cout + i]
        cout, a, b, c = int(cout), int(a), int(b), int(c)
        # 得到顶点位置
        a1, a2, a3 = n[a]
        b1, b2, b3 = n[b]
        c1, c2, c3 = n[c]
        # 绘制多面体
        scaleratio = 300
        glBegin(GL_POLYGON)
        glTexCoord2f( 0.0, 0.0)
        glVertex3f(a1 * scaleratio, a2 * scaleratio, a3 * scaleratio)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(b1 * scaleratio, b2 * scaleratio, b3 * scaleratio)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(c1 * scaleratio, c2 * scaleratio, c3 * scaleratio)
        glEnd();
    glPopMatrix()


    # left triangle
    vert = [0, 0, 0, 100, 100, 100]
    # right triangle
    vert2 = [200, 0, 200, 100, 300, 100]
    uv = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    #img.blit(200, 200)

    pyglet.gl.glDisable(tex.target)
    pyglet.graphics.draw(3, pyglet.gl.GL_TRIANGLES, ('v2f', tuple(vert2)))

    pyglet.gl.glEnable(tex.target)
    pyglet.gl.glBindTexture(tex.target, tex.id)
    pyglet.graphics.draw(3, pyglet.gl.GL_TRIANGLES, ('v2f', tuple(vert)), ('t2f', tuple(uv)))

pyglet.app.run()