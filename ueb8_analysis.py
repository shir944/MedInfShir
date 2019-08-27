# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:54:13 2017

@author: eschneid
http://pyopengl.sourceforge.net/context/tutorials/index.html
http://pyopengl.sourceforge.net/context/tutorials/shader_5.html
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import sys
import numpy as np
import traceback

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as QtCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class NPLFigure(QtCore.QObject, Figure):
    def __init__(self):
        super(NPLFigure, self).__init__()
        
class Figure(NPLFigure):
    def __init__(self):
        super.canvas = QtCanvas(self)
        
        self.ax1 = self.add_subplot(311)
        self.ax2 = self.add_subplot(312, sharex=self.ax1)
        self.ax3 = self.add_subplot(313, sharex=self.ax1)
        
        self.init_plots()
    
    def init_plots(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        
        def plot_data(slef):
            self.init_plots()
            #
            self.canvas.draw()
            
class AnalysisWidget(QtWidgets.QWidget):
    output_ready = QtCore.pyqtSignal(str)
    
    def __init__(self, *args, **kwargs):
        self.init_ui()
    
    def init_ui(self):
        self.fig1 = Figure1()
        self.toolbar = NavigationToolbar(self.fig1.canvas, self)
        self.btn_report = QtWidgets.QPushButton(self.tr('Report'))
        
        layout = QtWidgets.QBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.fig1.canva)
        layout.addWidget(self.btn_report)
        
    def plot_data(self, data):
        self.fig1.plot_data(data)
        
    def connect_save(self, fun):
        self.btn_report.clicked.connect(fun)
    
class Diagramm(Figure):
    def __init__(self, *args, **kwargs):
        super(Datendiagramm, self).__init__(*args, **kwargs)
        self.canvas = QtCanvas(self)
        
        self.ax = self.add_subplot(111)
        self.ax.grid()
        self.ax.set_xlabel('Frame #')
        self.ax.set_ylabel('Pupil Position [px]')
        
        self.refresh_counter = 0
        self.refresh_rate = 10
        self.lines = None
        
    def init(self, width, height, length):
        self.ax.cla()
        self.ax.grid()
        self.ax.set_xlabel('Frame #')
        self.ax.set_ylabel('Pupil Position [px]')
        self.ax.set_title('Pupil Center')
        
        self.ax.set_xlim([0, length])
        self.ax.set_ylim([0, np.maximum(width, height)])
        self.lines = self.ax.plot(np.full((length, 2), np.nan), linewidth=1)
        self.ax.legend(['Hor','Ver'])
        self.canvas.draw()
        
    """ Update the plot data for given index but don't draw """
    def set_pos(self, data):
        self.lines[0].set_ydata(data[:,0])
        self.lines[1].set_ydata(data[:,1])

    """ Redraw the plot """
    def refresh(self):
        if self.refresh_counter==0:
            if self.lines:
                self.ax.draw_artist(self.lines[0])
                self.ax.draw_artist(self.lines[1])
                self.canvas.update()
        self.refresh_counter = (self.refresh_counter+1)%self.refresh_rate
            # self.canvas.flush_events()
                           

class GLWidget(QtWidgets.QOpenGLWidget):

    shader = 0

    VERTEX_SHADER = """
        attribute vec3 Vertex_position;
        attribute vec3 Vertex_normal;
        varying vec3 baseNormal;
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * vec4(
                Vertex_position, 1.0
            );
            baseNormal = gl_NormalMatrix * normalize(Vertex_normal);
        }"""


    FRAGMENT_SHADER = """
        vec2 phong_weightCalc(
            in vec3 light_pos, // light position
            in vec3 half_light, // half-way vector between light and view
            in vec3 frag_normal, // geometry normal
            in float shininess
        ) {
            // returns vec2( ambientMult, diffuseMult )
            float n_dot_pos = max( 0.0, dot(
                frag_normal, light_pos
            ));
            float n_dot_half = 0.0;
            if (n_dot_pos > -.05) {
                n_dot_half = pow(max(0.0,dot(
                    half_light, frag_normal
                )), shininess);
            }
            return vec2( n_dot_pos, n_dot_half);
        }

        uniform vec4 Global_ambient;
        uniform vec4 Light_ambient;
        uniform vec4 Light_diffuse;
        uniform vec4 Light_specular;
        uniform vec3 Light_location;
        uniform float Material_shininess;
        uniform vec4 Material_specular;
        uniform vec4 Material_ambient;
        uniform vec4 Material_diffuse;
        varying vec3 baseNormal;
        void main() {
            // normalized eye-coordinate Light location
            vec3 EC_Light_location = normalize(
                gl_NormalMatrix * Light_location
            );
            // half-vector calculation
            vec3 Light_half = normalize(
                EC_Light_location - vec3( 0,0,-1 )
            );
            vec2 weights = phong_weightCalc(
                EC_Light_location,
                Light_half,
                baseNormal,
                Material_shininess
            );
            gl_FragColor = clamp(
            (
                (Global_ambient * Material_ambient)
                + (Light_ambient * Material_ambient)
                + (Light_diffuse * Material_diffuse * weights.x)
                // material's shininess is the only change here...
                + (Light_specular * Material_specular * weights.y)
            ), 0.0, 1.0);
        }
        """

    def __init__(self, **kwargs):
        super(GLWidget, self).__init__(**kwargs)

    def loadply(self, filename):
        from plyfile import PlyData, PlyElement
        ply = PlyData.read(filename)
        vertex = ply['vertex']
        coord = np.array([vertex['x'],vertex['y'],vertex['z'],
                          vertex['nx'],vertex['ny'],vertex['nz']],'f').transpose()
        
        tri_idx = ply['face']['vertex_indices']
        idx_dtype = tri_idx[0].dtype
        
        indices = np.fromiter(tri_idx, [('data', idx_dtype, (3,))],
                                   count=len(tri_idx))['data'].flatten()
        #indices = indices[0:9]
        print(coord)
        print(indices)
        return vbo.VBO(coord), vbo.VBO(indices, target=GL_ELEMENT_ARRAY_BUFFER), len(indices)

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(400, 400)

    def initializeGL(self):
        try:
            #self.coords,self.indices,self.count = Sphere(radius = 1).compile()
            # Compile shaders
            self.coord, self.indices, self.count = self.loadply('einfach.ply')
            self.stride = self.coord.data[0].nbytes

            vs = shaders.compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            self.shader = shaders.compileProgram(vs, fs)
            
            for uniform in (
                'Global_ambient',
                'Light_ambient','Light_diffuse','Light_location',
                'Light_specular',
                'Material_ambient','Material_diffuse',
                'Material_shininess','Material_specular',
            ):
                location = glGetUniformLocation( self.shader, uniform )
                if location in (None,-1):
                    print('Warning, no uniform: %s'%( uniform ))
                setattr( self, uniform+ '_loc', location )
            for attribute in (
                'Vertex_position','Vertex_normal',
            ):
                location = glGetAttribLocation( self.shader, attribute )
                if location in (None,-1):
                    print('Warning, no attribute: %s'%( uniform ))
                setattr( self, attribute+ '_loc', location )

            glClearColor(0, 0, 0, 1)
        except:
            traceback.print_exc()

    def resizeGL(self, width, height):
        try:
            self.width = width
            self.height = height
            h = float(height) / float(width);
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glFrustum(-1.0, 1.0, -h, h, 5.0, 60.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -25.0)
            glRotate( 20, 1,0,0 )
            self.update()
        except:
            traceback.print_exc()


    def paintGL(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            shaders.glUseProgram(self.shader)
            
            self.coord.bind()
            self.indices.bind()
            try:
                glUniform4f( self.Global_ambient_loc, .05,.05,.05,.1 )
                glUniform4f( self.Light_ambient_loc, .1,.1,.1, 1.0 )
                glUniform4f( self.Light_diffuse_loc, .25,.25,.25,1 )
                glUniform4f( self.Light_specular_loc, 0.0,1.0,0,1 )
                glUniform3f( self.Light_location_loc, 6,2,4 )
                glUniform4f( self.Material_ambient_loc, .1,.1,.1, 1.0 )
                glUniform4f( self.Material_diffuse_loc, .15,.15,.15, 1 )
                glUniform4f( self.Material_specular_loc, 1.0,1.0,1.0, 1.0 )
                glUniform1f( self.Material_shininess_loc, .95)
            
                glEnableVertexAttribArray( self.Vertex_position_loc )
                glEnableVertexAttribArray( self.Vertex_normal_loc )
                glVertexAttribPointer(
                    self.Vertex_position_loc,
                    3, GL_FLOAT,False, self.stride, self.coord
                )
                glVertexAttribPointer(
                    self.Vertex_normal_loc,
                    3, GL_FLOAT,False, self.stride, self.coord+(3*4)
                )
                #glDrawArrays(GL_TRIANGLES, 0, 18)
                glDrawElements(
                    GL_TRIANGLES, self.count,
                    GL_UNSIGNED_SHORT, self.indices
                )
            finally:
                self.coord.unbind()
                self.indices.unbind()
                glDisableVertexAttribArray( self.Vertex_position_loc )
                glDisableVertexAttribArray( self.Vertex_normal_loc )
        except:
            traceback.print_exc()
        finally:
            shaders.glUseProgram( 0 )

    def set_tweening(self, value):
        self.tween_fraction = value/100.0
        self.update()
        
    def idle(self):
        self.tween_fraction += 0.01
        if self.tween_fraction > 1.0:
            self.tween_fraction = 0.0
        self.update()

class MainWindow(QtWidgets.QWidget):
    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        self.init_ui()
        
        self.tabs = QtWidgets.QTabWidget()
        self.plot_widget = Diagramm()
        self.plot_widget.init(100,100,100)
        
    def init_ui(self):
        self.display = GLWidget(parent=self)
        
        # Timer und Timer-Funktion initialisieren
        #   self.refresh_timer = QtCore.QTimer(parent=self)
#        self.refresh_timer.timeout.connect(self.display.idle)
#        self.refresh_timer.start(1000.0 / 30)  # Refresh at 30Hz

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setTickInterval(1)
        self.slider.setValue(0)
        self.slider.setRange(0, 100)
#        self.slider.valueChanged.connect(self.display.set_tweening)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.display)
        layout.addWidget(self.slider)
        
        
    def refresh(self):
        try:
            self.display.update()
        except:
            traceback.print_exc()

if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.references = set()
    
    win = MainWindow()
    app.references.add(win)
    win.show()
    win.raise_()
    app.exec_()