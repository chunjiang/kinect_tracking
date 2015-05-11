import cv2 as cv
import numpy as np
import pykinectwindow as wxwindow
from OpenGL.GL import *
from OpenGL.GLU import *
import freenect
import calibkinect
import cProfile
from scipy.ndimage.filters import *
from threading import Thread, Lock
from time import sleep

###### OpenGL visualization courtesy of https://github.com/amiller/libfreenect-goodies

# I probably need more help with these!
try: 
    TEXTURE_TARGET = GL_TEXTURE_RECTANGLE
except:
    TEXTURE_TARGET = GL_TEXTURE_RECTANGLE_ARB

global depth, rgb, diff
if 'win' not in globals():
    global win
    win = wxwindow.Window(size=(640,480))


def refresh(): win.Refresh()

if not 'rotangles' in globals(): rotangles = [0,0]
if not 'zoomdist' in globals(): zoomdist = 1
if not 'projpts' in globals(): projpts = (None, None)
if not 'rgb' in globals():
    global rgb
    rgb = None

def create_texture():
    global rgbtex
    rgbtex = glGenTextures(1)
    glBindTexture(TEXTURE_TARGET, rgbtex)
    glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,GL_UNSIGNED_BYTE,None)


if not '_mpos' in globals(): _mpos = None
@win.eventx
def EVT_LEFT_DOWN(event):
    global _mpos
    _mpos = event.Position
  
@win.eventx
def EVT_LEFT_UP(event):
    global _mpos
    _mpos = None

@win.eventx
def EVT_MOTION(event):
    global _mpos
    if event.LeftIsDown():
        if _mpos:
            (x,y),(mx,my) = event.Position,_mpos
            rotangles[0] += y-my
            rotangles[1] += x-mx
            refresh()    
        _mpos = event.Position


@win.eventx
def EVT_MOUSEWHEEL(event):
    global zoomdist
    dy = event.WheelRotation
    zoomdist *= np.power(0.95, -dy / 8)
    refresh()
  

clearcolor = [0,0,0,0]
@win.event
def on_draw():  
    if not 'rgbtex' in globals():
        create_texture()

    xyz, uv = projpts
    if xyz is None: return

    if not rgb is None:
        rgb_ = (rgb.astype(np.float32) * 4 + 70).clip(0,255).astype(np.uint8)
        glBindTexture(TEXTURE_TARGET, rgbtex)
        glTexSubImage2D(TEXTURE_TARGET, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, rgb_);

    glClearColor(*clearcolor)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)

    # flush that stack in case it's broken from earlier
    glPushMatrix()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 4/3., 0.3, 200)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    def mouse_rotate(xAngle, yAngle, zAngle):
        glRotatef(xAngle, 1.0, 0.0, 0.0);
        glRotatef(yAngle, 0.0, 1.0, 0.0);
        glRotatef(zAngle, 0.0, 0.0, 1.0);
        
    glScale(zoomdist,zoomdist,1)
    glTranslate(0, 0,-3.5)
    mouse_rotate(rotangles[0], rotangles[1], 0);
    glTranslate(0,0,1.5)
    #glTranslate(0, 0,-1)

    # Draw some axes
    if 0:
        glBegin(GL_LINES)
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0)
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0)
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1)
        glEnd()

  # We can either project the points ourselves, or embed it in the opengl matrix
    if 0:
        print("this branch")
        dec = 4
        v,u = mgrid[:480,:640].astype(np.uint16)
        points = np.vstack((u[::dec,::dec].flatten(),
                            v[::dec,::dec].flatten(),
                            depth[::dec,::dec].flatten())).transpose()
        points = points[(points[:,2]<2047) & (points[:,2]>=0),:]
        
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glMultMatrixf(calibkinect.uv_matrix().transpose())
        glMultMatrixf(calibkinect.xyz_matrix().transpose())
        glTexCoordPointers(np.array(points))
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glMultMatrixf(calibkinect.xyz_matrix().transpose())
        glVertexPointers(np.array(points))
    else:
        glMatrixMode(GL_TEXTURE)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        #print(xyz[0])
        glVertexPointerf(xyz)
        glTexCoordPointerf(uv)

    # Draw the points
    glPointSize(2)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glEnable(TEXTURE_TARGET)
    glColor3f(1,1,1)
    glDrawElementsui(GL_POINTS, np.array(range(xyz.shape[0])))
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)
    glDisable(TEXTURE_TARGET)
    glPopMatrix()

    #
    if 0:
        inds = np.nonzero(xyz[:,2]>-0.55)
        glPointSize(10)
        glColor3f(0,1,1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawElementsui(GL_POINTS, np.array(inds))
        glDisableClientState(GL_VERTEX_ARRAY)

    if 0:
        # Draw only the points in the near plane
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glColor(0.9,0.9,1.0,0.8)
        glPushMatrix()
        glTranslate(0,0,-0.55)
        glScale(0.6,0.6,1)
        glBegin(GL_QUADS)
        glVertex3f(-1,-1,0); glVertex3f( 1,-1,0);
        glVertex3f( 1, 1,0); glVertex3f(-1, 1,0);
        glEnd()
        glPopMatrix()
        glDisable(GL_BLEND)

    glPopMatrix()

def make_colormap():
""" Make a 1024x3 matrix corresponding to a full walk around the color wheel. This is used for visualizing the range data.
"""
    global colormap
    defaultsize = (256,)
    
    # red high, green rising, blue low
    red = 255 * np.ones(defaultsize)
    green = np.arange(256)
    blue = np.zeros(defaultsize)
    
    #red falling, green high, blue low
    red = np.hstack((red, 255 - np.arange(256)))
    green = np.hstack((green, 255 * np.ones(defaultsize)))
    blue = np.hstack((blue, np.zeros(defaultsize)))
    
    #red low, green high, blue rising
    red = np.hstack((red, np.zeros(defaultsize)))
    green = np.hstack((green, 255 * np.ones(defaultsize)))
    blue = np.hstack((blue, np.arange(256)))
    
    #red low, green falling, blue high
    red = np.hstack((red, np.zeros(defaultsize)))
    green = np.hstack((green, 255 - np.arange(256)))
    blue = np.hstack((blue, 255 * np.ones(defaultsize)))
    
    #red rising, green low, blue high
    red = np.hstack((red, np.arange(256)))
    green = np.hstack((green, np.zeros(defaultsize)))
    blue = np.hstack((blue, 255 * np.ones(defaultsize)))
    
    #red high, green low, blue falling
    red = np.hstack((red, 255 * np.ones(defaultsize)))
    green = np.hstack((green, np.zeros(defaultsize)))
    blue = np.hstack((blue, 255 - np.arange(256)))

    colormap = np.vstack((red, green, blue)).astype(np.uint8).transpose()

def depth_to_rgb(d):
""" Take a MxNx1 depth matrix and convert it to a MxNx3 RGB color matrix.
"""
    # Represent 0 data and 2047 data as black and white respectively.
    zeros = d == 0
    maxs = d == 2047
    d[maxs] = 0
    whiteval = np.array([0, 0, 0]).astype(np.uint8)
    zeroval = np.array([255, 255, 255]).astype(np.uint8)
    
    # Find the min and max real data for use in scaling the color scheme
    if not np.any(d != 2047):
        r,c = d.shape
        return np.zeros((r,c,3)) # All data 2047 so return all white
    top = np.amax(d[d != 2047])
    if not np.any(d):
        r,c = d.shape
        return 255 * np.ones((r,c,3)) # All data 0 so return all black
    bot = np.amin(d[np.nonzero(d)])
    
    
    dint = d.astype(np.int)
    d = ((dint - bot) * 1024) / (top - bot) # linearly scales the range of input data to 0-1024
    d[d < 0] = 0 # Subtraction makes some values negative. Correct before mapping depth to colors
    colors = colormap[d]
    colors[zeros] = zeroval
    colors[maxs] = whiteval
    
    return colors
    
def sample_bgnd(n):
""" Take frames from the kinect to use as the background. Argument n
corresponds to number of frames to be averaged as background. Currently
averaging hurts more than it helps.
"""
    depth, _ = freenect.sync_get_depth()
    #bgnd = uniform_filter(depth, mode='constant')
    bgnd = depth
    
    for _ in range(n - 1):
        depth, _ = freenect.sync_get_depth()       
        #bgnd += uniform_filter(depth, mode='constant')
        bgnd += depth
    bgnd = np.around(bgnd / float(n)).astype(np.int)
    print("Got Background")
    return bgnd
    
def warmup(time):
""" At startup, the kinect depth sensor needs to go through some range
calibration and focusing. This function accepts number of seconds to wait
and will take some frames to "warm up" the sensor.
"""
    dt = .05
    iterations = int(np.around(time / dt))
    for i in range(iterations):
        freenect.sync_get_depth()
        if i % 20 == 0:
            print("Warming up " + str(i/20))
        sleep(dt)

def make_windows():
""" Just makes openCV windows. Only useful while debugging.
"""
    #cv.namedWindow('Background')
    #cv.namedWindow('Current')
    #cv.namedWindow('Diff')
    cv.waitKey(5)

try:

    make_colormap()
    make_windows()
    warmup(10)              # wait 10 sec before collecting background data
    bgnd = sample_bgnd(1)   # 1 sample is most effective
    #cv.imshow('Background', depth_to_rgb(bgnd))
    cv.waitKey(5)
    
    d = 2                   #downsample factor
    X,Y = np.meshgrid(range(640),range(480))
    X = X[::d,::d]
    Y = Y[::d,::d]
    bgnd = bgnd[::d,::d]
    
    # Start a new thread to handle the 3D pointcloud visualization
    global opengl_thread, stop
    stop = False
    def run_opengl():
        while not stop:
            refresh()
            sleep(.02)

    opengl_thread = Thread(target=run_opengl)
    started = False
    
    # Keep track of nonzero data in a previous frame.
    prev = bgnd.copy()
    prev[prev.nonzero()] = 1
    
    cloud = None # Variable to keep track of center of mass
    # Image pipeline:
    # 1. Downsample depth data
    # 2. Disregard depth points that differ by less than 10 from the
    #    background
    # 3. Progressively larger median filter. Start with 3x3, up to 4x4
    #    This helps to eliminate small blobs on the difference image
    # 4. Filter transients by disregarding nonzero values in the current
    #    difference frame that are zero in the previous difference frame
    # 5. Run the data through some specially formulated matrices to convert
    #    (row, column, depth) to (x, y, z) in meters. This formula courtesy
    #    of https://github.com/amiller/libfreenect-goodies as well.
    # 6. If there are more than 15 points (Object introduced into image),
    #    compute the average of all the x, y, and z values to give the
    #    centroid.
    while True:
        depth, _ = freenect.sync_get_depth()
        rgb,_ = freenect.sync_get_video()
        depth = depth[::d, ::d]
        intdepth = depth.astype(np.int)
        
        diff = intdepth
        diff[np.fabs(intdepth - bgnd) < 10] = 0
        for i in range(3,4):
            diff = median_filter(diff, size=(i,i))
        
        old_diff = diff.copy()
        nonz_idx = diff.nonzero()
        diff[nonz_idx] = prev[nonz_idx] * diff[nonz_idx]
        
        projpts = calibkinect.depth2xyzuv(diff, X, Y)
        points = projpts[0]
        if points.shape[0] < 15:
            cloud = None
        else:
            cloud = np.mean(points, axis=0)
            print(cloud)
        #cv.imshow('Current', depth_to_rgb(depth))
        cv.imshow('Diff', depth_to_rgb(diff))
        cv.waitKey(5)
        prev = old_diff
        prev[prev.nonzero()] = 1
        
        #print("Projected points")
        #refresh()
        #print("Drew image")

        if not started:
            started = True
            opengl_thread.start()
            print("Started refresh thread")
        #break
        sleep(.02)

except KeyboardInterrupt:
    pass
finally:
    stop = True
    print("stopping")


