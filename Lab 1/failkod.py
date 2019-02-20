import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d  as conv2

def image_gradient(J, I, ksize, sigma):
#    g = np.zeros(ksize,ksize)
#    g[ksize//2, ksize//2] = 1
#    g = scipy.ndimage.filters.gaussian_filter(g,sigma)
    (x,y) = np.meshgrid(np.arange(-(ksize-1)/2,(ksize-1)/2+1,1),np.arange(-(ksize-1)/2,(ksize-1)/2+1,1))
    g = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
    gdx = -(x/sigma**2)*g
    gdy = -(y/sigma**2)*g
    Ig = conv2(I, g, mode='same')
    Jg = conv2(J, g, mode='same')
    Jgdx = conv2(J, gdx, mode='same')
    Jgdy = conv2(J, gdy, mode='same')
    return (Ig, Jg, Jgdx, Jgdy)

def estimate_T(Jgdx, Jgdy, rows, cols, window_size):
    T = np.zeros((2,2))
    for i in range(0, int(window_size[0]-1)):
        for j in range(0, int(window_size[1]-1)):
            T = T + np.array([ [Jgdx[i,j]**2, Jgdx[i,j]*Jgdy[i,j] ], [ Jgdx[i,j]*Jgdy[i,j], Jgdy[i,j]**2 ]])

    return T

def estimate_e(Jg, Ig, Jgdx, Jgdy, rows, cols, window_size):
    e = np.zeros((2,1))
    for i in range(0, int(window_size[0]-1)):
        for j in range(0, int(window_size[1]-1)):
            e = e + (Ig[i,j]-Jg[i,j])*np.transpose(np.array([[ Jgdx[i,j], Jgdy[i,j] ]]))

    return e


def myInterpolation(spline, d, point, window_size):
    rows = np.arange(point[0]-(window_size[0]-1)/2, point[0]+(window_size[0]-1)/2,1)
    cols = np.arange(point[1]-(window_size[1]-1)/2, point[1]+(window_size[1]-1)/2,1)
    rows -= d[0]
    cols -= d[1]
    interpolatedIm = spline(rows, cols)
    return interpolatedIm

def getSpline(Im):
    x = np.arange(0, np.size(Im,1) ,1)
    y = np.arange(0, np.size(Im,0), 1)
    intSpline = scipy.interpolate.RectBivariateSpline(y, x, Im)
    return intSpline

def myKLTracker(J, I, point, window_size=np.transpose(np.array([71,41])), maxIterations=100):
    dTot = np.array([[0.0],[0.0]])
    d = np.transpose(np.array([np.Inf, np.Inf]))
    i = 1

    ksize = 5;
    sigma = 1;
    (Ig, Jg, Jgdx, Jgdy) = image_gradient(J, I, ksize, sigma)

    JgSpline = getSpline(Jg)
    JgdxSpline = getSpline(Jgdx)
    JgdySpline = getSpline(Jgdy)

    #rows = int(point[0]-(window_size[0]-1)/2):int(point[0]+(window_size[0]-1)/2)
    #cols = int(point[1]-(window_size[1]-1)/2):int(point[1]+(window_size[1]-1)/2)
    #rint(np.shape(rows))
    #print(np.shape(cols))
    #print(np.shape(Ig))
    IgWarp = Ig[int(point[0]-(window_size[0]-1)/2):int(point[0]+(window_size[0]-1)/2),int(point[1]-(window_size[1]-1)/2):int(point[1]+(window_size[1]-1)/2)]
    while i <= maxIterations and np.linalg.norm(d) > 0.1 :

        JgWarp = myInterpolation(JgSpline, dTot, point, window_size)
        JgdxWarp = myInterpolation(JgdxSpline, dTot, point, window_size)
        JgdyWarp = myInterpolation(JgdySpline, dTot, point, window_size)

        T = estimate_T(JgdxWarp, JgdyWarp, point[0], point[1], window_size)
        e = estimate_e(JgWarp, IgWarp, JgdxWarp, JgdyWarp, point[0], point[1], window_size)
        d = np.linalg.solve(T, e)
        dTot += d
        print(dTot)
        i += 1
    return dTot




import matplotlib.image as mpimage
im = mpimage.imread("flower.jpeg")
J = im[:,:,1].astype(float)
dReal = np.transpose(np.array([20.2, 20.2]))
JSpline = getSpline(J)

rows = np.arange(0.0,499.0,1)
cols = np.arange(0.0,330.0,1)
rows -= dReal[0]
cols -= dReal[1]
I = JSpline(rows, cols)


d = myKLTracker(J, I, np.array([150,150]))

print(d)
