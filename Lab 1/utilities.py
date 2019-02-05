import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d  as conv2

def image_gradient(I, J, ksize, sigma):
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

def estimate_T(Jgdx, Jgdy, x, y, window_size):
    T = np.zeros((2,2))
    for i in range(int(x-(window_size-1)/2), int(x+(window_size-1)/2)):
        for j in range(int(y-(window_size-1)/2), int(y+(window_size-1)/2)):
            T = T + np.array([ [Jgdx[i,j]**2, Jgdx[i,j]*Jgdy[i,j] ], [ Jgdx[i,j]*Jgdy[i,j], Jgdy[i,j]**2 ]])

    return T

def estimate_e(Ig, Jg, Jgdx, Jgdy, x, y, window_size):
    e = np.zeros((2,1))
    for i in range(int(x-(window_size-1)/2), int(x+(window_size-1)/2)):
        for j in range(int(y-(window_size-1)/2), int(y+(window_size-1)/2)):
            T = T + (I[i,j]-J[i,j])*np.array([[]])

    return e





import matplotlib.image as mpimage
I = mpimage.imread("flower.jpeg")
I = I[:,:,1]
J = I
(Ig, Jg, Jgdx, Jgdy) = image_gradient(I, J, 5, 1)
T = estimate_T(Jgdx, Jgdy, 5, 5, 3)
print(T)
