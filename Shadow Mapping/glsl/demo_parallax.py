#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Pyglet GLSL Demo Dot3 Bumpmap Shader on http://www.pythonstuff.org
# pythonstuff_at_gmx_dot_at  (c) 2010
#
# based on the "graphics.py" batch/VBO demo by
# pyglet
# Copyright (c) 2006-2008 Alex Holkner
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#  * Neither the name of pyglet nor the names of its
#    contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ----------------------------------------------------------------------------

'''This expands the previous example by parallax by heightfield mapping the bumpmapped texture
   Find more GLSL-examples http://www.pythonstuff.org
'''
html = '''
<font size=+3 color=#FF3030>
<b>Pyglet GLSL Parallax Mapping Demo</b>
</font><br/>
<font size=+2 color=#00FF60>
P, O = Parallax inc/dec<br/>
B = Bumpmap on/off<br/>
ENTER = Shader on/off<br/>
R = Reset<br/>
Q, Esc = Quit<br/>
F = Toggle Figure<br/>
T = Toggle Texture<br/>
W, S, A, D = Up, Down, Left, Right<br/>
Space = Move/Stop<br/>
Arrows = Move Light 0<br/>
H = This Help<br/>
</font>
'''

from math import pi, sin, cos, sqrt
from euclid import *

import pyglet
from pyglet.gl import *
from pyglet.window import key
from pyglet import image, resource

from shader import Shader

resource.path.append('textures')
resource.reindex()
texturecnt = 3  # Texturemap0.jpg = Colormap Texturemap1.jpg = Bumpmap Texturemap2.jpg = Heightmap

try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True, )
    window = pyglet.window.Window(resizable=True, config=config, vsync=False)  # "vsync=False" to check the framerate
except pyglet.window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = pyglet.window.Window(resizable=True)

label = pyglet.text.HTMLLabel(html,  # location=location,
                              width=window.width // 2,
                              multiline=True, anchor_x='center', anchor_y='center')

fps_display = pyglet.clock.ClockDisplay()  # see programming guide pg 48


@window.event
def on_resize(width, height):
    if height == 0: height = 1
    # Keep text vertically centered in the window
    label.y = window.height // 2
    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED


def update(dt):
    global autorotate
    global rot
    global dist

    if autorotate:
        rot += Vector3(0.1, 12, 5) * dt
        rot.x %= 360
        rot.y %= 360
        rot.z %= 360


pyglet.clock.schedule(update)


def dismiss_dialog(dt):
    global showdialog
    showdialog = False


pyglet.clock.schedule_once(dismiss_dialog, 10.0)


# Define a simple function to create ctypes arrays of floats:
def vec(*args):
    return (GLfloat * len(args))(*args)


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glLoadIdentity()
    glTranslatef(0.0, 0.0, dist);
    glRotatef(rot.x, 0, 0, 1)
    glRotatef(rot.y, 0, 1, 0)
    glRotatef(rot.z, 1, 0, 0)

    glPolygonMode(GL_FRONT, GL_FILL)

    if shaderon:
        # bind our shader
        shader.bind()
        shader.uniformi('toggletexture', toggletexture)
        shader.uniformi('togglebump', togglebump)
        shader.uniformf('parallaxheight', parallaxheight)
        for i in range(texturecnt):
            glActiveTexture(GL_TEXTURE0 + i)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture[i].id)
            shader.uniformi('my_color_texture[' + str(i) + ']', i)
        if togglefigure:
            batch1.draw()
        else:
            batch2.draw()

        for i in range(texturecnt):
            glActiveTexture(GL_TEXTURE0 + i)
            glDisable(GL_TEXTURE_2D)
        shader.unbind()
    else:
        if togglefigure:
            batch1.draw()
        else:
            batch2.draw()

    glActiveTexture(GL_TEXTURE0)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    if showdialog:
        glLoadIdentity()
        glTranslatef(0, -200, -450)
        label.draw()

    glLoadIdentity()
    glTranslatef(250, -290, -500)
    fps_display.draw()

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glDisable(GL_TEXTURE_2D)


@window.event
def on_key_press(symbol, modifiers):
    global autorotate
    global rot
    global dist
    global togglefigure
    global toggletexture
    global togglebump
    global parallaxheight
    global light0pos
    global light1pos
    global showdialog
    global shaderon

    if symbol == key.R:
        print
        'Reset'
        rot = Vector3(0, 0, 0)
    elif symbol == key.ESCAPE or symbol == key.Q:
        print
        'Good Bye !'  # ESC would do it anyway, but not "Q"
        pyglet.app.exit()
        return pyglet.event.EVENT_HANDLED
    elif symbol == key.H:
        showdialog = not showdialog
    elif symbol == key.ENTER:
        print
        'Shader toggle'
        shaderon = not shaderon
    elif symbol == key.SPACE:
        print
        'Toggle autorotate'
        autorotate = not autorotate
    elif symbol == key.F:
        togglefigure = not togglefigure
        print
        'Toggle Figure ', togglefigure
    elif symbol == key.B:
        togglebump = not togglebump
        print
        'Toggle Bumpmap ', togglebump
    elif symbol == key.P:
        parallaxheight += 0.01
        print
        'Parallax Height now ', parallaxheight
    elif symbol == key.O:
        parallaxheight -= 0.01
        if parallaxheight <= 0.0:
            parallaxheight = 0.0
            print
            'Parallax now OFF'
        else:
            print
            'Parallax Height now ', parallaxheight
    elif symbol == key.PLUS:
        dist += 0.5
        print
        'Distance now ', dist
    elif symbol == key.MINUS:
        dist -= 0.5
        print
        'Distance now ', dist
    elif symbol == key.T:
        toggletexture = not toggletexture
        print
        'Toggle Texture ', toggletexture
    elif symbol == key.A:
        print
        'Stop left'
        if autorotate:
            autorotate = False
        else:
            rot.y += -rotstep
            rot.y %= 360
    elif symbol == key.S:
        print
        'Stop down'
        if autorotate:
            autorotate = False
        else:
            rot.z += rotstep
            rot.z %= 360
    elif symbol == key.W:
        print
        'Stop up'
        if autorotate:
            autorotate = False
        else:
            rot.z += -rotstep
            rot.z %= 360
    elif symbol == key.D:
        print
        'Stop right'
        if autorotate:
            autorotate = False
        else:
            rot.y += rotstep
            rot.y %= 360
    elif symbol == key.LEFT:
        print
        'Light0 rotate left'
        tmp = light0pos[0]
        light0pos[0] = tmp * cos(lightstep) - light0pos[2] * sin(lightstep)
        light0pos[2] = light0pos[2] * cos(lightstep) + tmp * sin(lightstep)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.RIGHT:
        print
        'Light0 rotate right'
        tmp = light0pos[0]
        light0pos[0] = tmp * cos(-lightstep) - light0pos[2] * sin(-lightstep)
        light0pos[2] = light0pos[2] * cos(-lightstep) + tmp * sin(-lightstep)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.UP:
        print
        'Light0 up'
        tmp = light0pos[1]
        light0pos[1] = tmp * cos(-lightstep) - light0pos[2] * sin(-lightstep)
        light0pos[2] = light0pos[2] * cos(-lightstep) + tmp * sin(-lightstep)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    elif symbol == key.DOWN:
        print
        'Light0 down'
        tmp = light0pos[1]
        light0pos[1] = tmp * cos(lightstep) - light0pos[2] * sin(lightstep)
        light0pos[2] = light0pos[2] * cos(lightstep) + tmp * sin(lightstep)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    else:
        print
        'OTHER KEY'


def setup():
    # One-time GL setup
    global light0pos
    global light1pos
    global toggletexture
    global togglebump
    global parallaxheight
    global texture

    light0pos = [20.0, 20.0, 20.0, 1.0]  # positional light !
    light1pos = [-20.0, -20.0, 20.0, 0.0]  # infinitely away light !

    glClearColor(1, 1, 1, 1)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)

    texture = []
    for i in range(texturecnt):
        texturefile = 'Texturemap' + str(i) + '.jpg'
        print
        "Loading Texture", texturefile
        textureSurface = pyglet.resource.texture(texturefile)
        texture.append(textureSurface.get_texture())
        glBindTexture(texture[i].target, texture[i].id)
        print
        "Texture ", i, " bound to ", texture[i].id

    # Uncomment this line for a wireframe view
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Simple light setup.  On Windows GL_LIGHT0 is enabled by default,
    # but this is not the case on Linux or Mac, so remember to always
    # include it.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
    glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.9, 0.9, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

    glLightfv(GL_LIGHT1, GL_POSITION, vec(*light1pos))
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.6, .6, .6, 1.0))
    glLightfv(GL_LIGHT1, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.5, 0.5, 0.5, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)


# create our Phong Shader by Jerome GUINOT aka 'JeGX' - jegx [at] ozone3d [dot] net
# see http://www.ozone3d.net/tutorials/glsl_lighting_phong.php

shader = Shader(['''
varying vec3 lightDir0, lightDir1, eyeVec;
varying vec3 normal, tangent, binormal;

void main()
{
// Create the Texture Space Matrix
	normal   = normalize(gl_NormalMatrix * gl_Normal);
        tangent  = normalize(gl_NormalMatrix * (gl_Color.rgb - 0.5));
        binormal = cross(normal, tangent);
   mat3 TBNMatrix = mat3(tangent, binormal, normal);

	vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

	lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex) * TBNMatrix;
	lightDir1 = vec3(gl_LightSource[1].position.xyz - vVertex) * TBNMatrix;
	eyeVec    = -vVertex * TBNMatrix;

	gl_Position = ftransform();
        gl_TexCoord[0]  = gl_TextureMatrix[0] * gl_MultiTexCoord0;
}
'''], ['''
varying vec3 normal, lightDir0, lightDir1, eyeVec;
uniform sampler2D my_color_texture[''' + str(texturecnt) + ''']; //0 = ColorMap, 1 = NormalMap, 2 = HeightMap
uniform int   toggletexture;  // false/true
uniform int   togglebump;     // false/true
uniform float parallaxheight;

void main (void)
{
// Compute parallax displaced texture coordinates
  vec3 eye = normalize(-eyeVec);
  vec2 offsetdir = vec2( eye.x, eye.y );
  vec2 coords1 = gl_TexCoord[0].st;
  float height1 = parallaxheight * (texture2D( my_color_texture[2], coords1).r - 0.5);
  vec2 offset1  = height1 * offsetdir;
  vec2 coords2  = coords1 + offset1;
  float height2 = parallaxheight * (texture2D( my_color_texture[2], coords2).r - 0.5);
//vec2 offset2  = height2 * offsetdir;
  vec2 newCoords = coords2;
  if ( length( offset1 ) > 0.001 ) // 5.0 * abs( height1 ) > abs( height2 ) )
    newCoords = coords1 + (height2/height1) * offset1;


        vec4 texColor = vec4(texture2D(my_color_texture[0], newCoords).rgb, 1.0);
        vec3 norm     = normalize( texture2D(my_color_texture[1], newCoords).rgb - 0.5);

        if ( toggletexture == 0 ) texColor = gl_FrontMaterial.ambient;
        vec4 final_color = (gl_FrontLightModelProduct.sceneColor * vec4(texColor.rgb,1.0)) +
	(gl_LightSource[0].ambient * vec4(texColor.rgb,1.0)) +
	(gl_LightSource[1].ambient * vec4(texColor.rgb,1.0));

	vec3 N = (togglebump != 0) ? normalize(norm) : vec3(0.0, 0.0, 1.0 );
	vec3 L0 = normalize(lightDir0);
	vec3 L1 = normalize(lightDir1);

	float lambertTerm0 = dot(N,L0);
	float lambertTerm1 = dot(N,L1);

	if(lambertTerm0 > 0.0)
	{
		final_color += gl_LightSource[0].diffuse *
		               gl_FrontMaterial.diffuse *
					   lambertTerm0;

		vec3 E = normalize(eyeVec);
		vec3 R = reflect(-L0, N);
		float specular = pow( max(dot(R, E), 0.0),
		                 gl_FrontMaterial.shininess );
		final_color += gl_LightSource[0].specular *
		               gl_FrontMaterial.specular *
					   specular;
	}
	if(lambertTerm1 > 0.0)
	{
		final_color += gl_LightSource[1].diffuse *
		               gl_FrontMaterial.diffuse *
					   lambertTerm1;

		vec3 E = normalize(eyeVec);
		vec3 R = reflect(-L1, N);
		float specular = pow( max(dot(R, E), 0.0),
		                 gl_FrontMaterial.shininess );
		final_color += gl_LightSource[1].specular *
		               gl_FrontMaterial.specular *
					   specular;
	}
	gl_FragColor = final_color;
}
'''])


class Torus(object):
    list = None

    def __init__(self, radius, inner_radius, slices, inner_slices,
                 batch, group=None):
        # Create the vertex and normal arrays.
        vertices = []
        normals = []
        textureuvw = []
        tangents = []

        u_step = 2 * pi / (slices - 1)
        v_step = 2 * pi / (inner_slices - 1)
        u = 0.
        for i in range(slices):
            cos_u = cos(u)
            sin_u = sin(u)
            v = 0.
            for j in range(inner_slices):
                cos_v = cos(v)
                sin_v = sin(v)

                d = (radius + inner_radius * cos_v)
                x = d * cos_u
                y = inner_radius * sin_v
                z = -d * sin_u

                nx = cos_u * cos_v
                ny = sin_v
                nz = -sin_u * cos_v

                n = sqrt(nx * nx + ny * ny + nz * nz)
                if n < 0.99 or n > 1.01:
                    nx = nx / n
                    ny = ny / n
                    nz = nz / n
                    print
                    "Torus: N normalized"

                tx = -sin_u
                ty = 0
                tz = -cos_u

                a = sqrt(tx * tx + ty * ty + tz * tz)
                if a > 0.001:
                    tx = tx / a
                    ty = ty / a
                    tz = tz / a

                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
                textureuvw.extend([u / (2.0 * pi), v / (2.0 * pi), 0.0])
                tangents.extend([int(round(255 * (0.5 - 0.5 * tx))),
                                 int(round(255 * (0.5 - 0.5 * ty))),
                                 int(round(255 * (0.5 - 0.5 * tz)))])
                v += v_step
            u += u_step

        # Create a list of triangle indices.
        indices = []
        for i in range(slices - 1):
            for j in range(inner_slices - 1):
                p = i * inner_slices + j
                indices.extend([p, p + inner_slices, p + inner_slices + 1])
                indices.extend([p, p + inner_slices + 1, p + 1])

        self.vertex_list = batch.add_indexed(len(vertices) // 3,
                                             GL_TRIANGLES,
                                             group,
                                             indices,
                                             ('v3f/static', vertices),
                                             ('n3f/static', normals),
                                             ('t3f/static', textureuvw),
                                             ('c3B/static', tangents))

    def delete(self):
        self.vertex_list.delete()


class Sphere(object):
    list = None

    def __init__(self, radius, slices, batch, group=None):
        # Create the vertex and normal arrays.
        vertices = []
        normals = []
        textureuvw = []
        tangents = []

        u_step = 2 * pi / (slices - 1)
        v_step = pi / (slices - 1)
        u = 0.
        for i in range(slices):
            cos_u = cos(u)
            sin_u = sin(u)
            v = 0.
            for j in range(slices):
                cos_v = cos(v)
                sin_v = sin(v)

                nx = sin_v * cos_u
                ny = -cos_v
                nz = -sin_v * sin_u

                n = sqrt(nx * nx + ny * ny + nz * nz)
                if n < 0.99 or n > 1.01:
                    nx = nx / n
                    ny = ny / n
                    nz = nz / n
                    print
                    "Sphere: N normalized"

                tx = nz
                ty = 0
                tz = -nx

                a = sqrt(tx * tx + ty * ty + tz * tz)
                if a > 0.001:
                    tx = tx / a
                    ty = ty / a
                    tz = tz / a

                x = radius * nx
                y = radius * ny
                z = radius * nz

                vertices.extend([x, y, z])
                normals.extend([nx, ny, nz])
                textureuvw.extend([u / (2 * pi), v / (pi), 0.0])
                tangents.extend([int(round(255 * (0.5 - 0.5 * tx))),
                                 int(round(255 * (0.5 - 0.5 * ty))),
                                 int(round(255 * (0.5 - 0.5 * tz)))])
                v += v_step
            u += u_step

        # Create a list of triangle indices.
        indices = []
        for i in range(slices - 1):
            for j in range(slices - 1):
                p = i * slices + j
                indices.extend([p, p + slices, p + slices + 1])
                indices.extend([p, p + slices + 1, p + 1])

        self.vertex_list = batch.add_indexed(len(vertices) // 3,
                                             GL_TRIANGLES,
                                             group,
                                             indices,
                                             ('v3f/static', vertices),
                                             ('n3f/static', normals),
                                             ('t3f/static', textureuvw),
                                             ('c3B/static', tangents))

    def delete(self):
        self.vertex_list.delete()


dist = -3.5
rot = Vector3(0, 0, 0)
autorotate = True
rotstep = 10
lightstep = 10 * pi / 180
togglefigure = False
toggletexture = True
togglebump = True
parallaxheight = 0.02
showdialog = True
shaderon = True

setup()

batch1 = pyglet.graphics.Batch()
torus = Torus(1, 0.3, 80, 25, batch=batch1)
batch2 = pyglet.graphics.Batch()
sphere = Sphere(1.2, 50, batch=batch2)
pyglet.app.run()

# thats all