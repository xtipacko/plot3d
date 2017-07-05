import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


class Plot:
    def __init__(self, wxyz):
        self.wxyz = wxyz
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['toolbar'] = 'None'
        self.fig = plt.figure()
        self.fig.suptitle('Linear Regression', fontsize=14, color=(0,0.6,0))
        self.fig.set_facecolor((0,0,0))
        self.ax = self.fig.gca(projection='3d',aspect='equal')
        #ax.set_title()
        self.ax.title.set_color((0.1,0.7,0))
        self.ax.set_facecolor('black')
        self.ax.grid(b=False)
        #setting middle button to scale plot
        self.ax.mouse_init(zoom_btn=2)
        # Get rid of the panes (kinda boxes)
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # self.ax.tick_params(axis='x', colors='white')
        # self.ax.tick_params(axis='y', colors='white')
        # self.ax.tick_params(axis='z', colors='white')
        # Get rid of the spines
        self.ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # Draw Axes
        self.drawXYZ(*(self.wxyz*1.8),
                xcolor=(1,0.7,0.7), 
                ycolor=(0.7,1,0.7),
                zcolor=(0.7,0.7,1))

        self.ax.set_xlim([-1*self.wxyz[0], self.wxyz[0]])
        self.ax.set_ylim([-1*self.wxyz[1], self.wxyz[1]])
        self.ax.set_zlim([-1*self.wxyz[2], self.wxyz[2]])
        
    def drawXYZ(self, XW,YW,ZW, xcolor=(1,1,1), ycolor=(1,1,1), zcolor=(1,1,1)):
        xcoord = np.array([-1, 0, 0, 2, 0, 0])*XW
        ycoord = np.array([0, -1, 0, 0, 2, 0])*YW
        zcoord = np.array([0, 0, -1, 0, 0, 2])*ZW
        #X, Y, Z, U, V, W = coord x,y,z - start_position; u,v,w - vector components
        # DRAWS X,Y,Z AXES:
        self.ax.quiver(*xcoord, length=1, arrow_length_ratio=0.05, color=xcolor) 
        self.ax.quiver(*ycoord, length=1, arrow_length_ratio=0.05, color=ycolor)
        self.ax.quiver(*zcoord, length=1, arrow_length_ratio=0.05, color=zcolor)
        #LABELS THEM
        self.ax.text(XW+0.1, 0, 0, "X", color=xcolor)
        self.ax.text(0, YW+0.1, 0, "Y", color=ycolor)
        self.ax.text(0, 0, ZW+0.1, "Z", color=zcolor)

    def show(self):
        self.ax.legend()
        plt.show()


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin


if __name__ == '__main__':
    plot = Plot(np.array([1,1,1]))
    n = 100    
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for c, m, zlow, zhigh in [('b', 'o', -1, 1), ('g', 's', 1, -1)]:
        xs = randrange(n, -1, 1)
        ys = randrange(n, -1, 1)
        zs = randrange(n, zlow, zhigh)
        plot.ax.scatter(xs, ys, zs, c=c, marker=m)   
    #plane plotting
    point  = np.array([0, 0, 0])
    normal = np.array([1, 1, 6])
    d = -point.dot(normal)
    xx, yy = np.meshgrid(np.linspace(-1, 1, num=10), np.linspace(-1, 1, num=10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]    
    ln = plot.ax.plot_surface(xx, yy, z, color=(0.3,0.7,1,0.5),shade=False) 
    plt.pause(5)   
    ln.remove()
    plot.show()