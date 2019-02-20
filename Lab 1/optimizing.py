import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d  as conv2

def lowpassFilter(Im, ksize, sigma):
    (x,y) = np.meshgrid(np.arange(-(ksize-1)/2,(ksize-1)/2+1,1),np.arange(-(ksize-1)/2,(ksize-1)/2+1,1))
    g = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
    Img = conv2(Im, g, mode='same')
    return Img


def image_gradient(Im, ksize, sigma):
    (x,y) = np.meshgrid(np.arange(-(ksize-1)/2,(ksize-1)/2+1,1),np.arange(-(ksize-1)/2,(ksize-1)/2+1,1))
    g = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
    gdx = -(x/sigma**2)*g
    gdy = -(y/sigma**2)*g
    Imdx = conv2(Im, gdx, mode='same')
    Imdy = conv2(Im, gdy, mode='same')
    return (Imdx, Imdy)

def getTensorData(Jgdx, Jgdy):
    imShape = np.shape(Jgdx)
    Jgdx2 = np.zeros(imShape)
    Jgdy2 = np.zeros(imShape)
    Jgdxdy = np.zeros(imShape)
    for i in range(0,imShape[0]-1):
        for j in range(0,imShape[1]-1):
            Jgdx2[i,j] = Jgdx[i,j]**2
            Jgdy2[i,j] = Jgdy[i,j]**2
            Jgdxdy[i,j] = Jgdx[i,j]*Jgdy[i,j]
    return (Jgdx2, Jgdy2, Jgdxdy)


def estimate_T(Jgdx2, Jgdy2, Jgdxdy, x, y, window_size):
    T = np.zeros((2,2))
    xMax = np.size(Jgdx2,1)-1
    yMax = np.size(Jgdx2,0)-1

    for i in range(int(y-(window_size[1]-1)/2), int(y+(window_size[1]-1)/2)):
        if i < 0:
            i = 0
        elif i > yMax:
            i = yMax
        for j in range(int(x-(window_size[0]-1)/2), int(x+(window_size[0]-1)/2)):
            if j < 0:
                j = 0
            elif j > xMax:
                j = xMax
            T = T + np.array([ [Jgdx2[i,j], Jgdxdy[i,j] ], [ Jgdxdy[i,j], Jgdy2[i,j] ]])
    return T

def estimate_e(Jg, Ig, Jgdx, Jgdy, x, y, window_size):
    e = np.zeros((2,1))
    for i in range(int(y-(window_size[1]-1)/2), int(y+(window_size[1]-1)/2)):
        for j in range(int(x-(window_size[0]-1)/2), int(x+(window_size[0]-1)/2)):
            e = e + (Ig[i,j]-Jg[i,j])*np.transpose(np.array([[ Jgdx[i,j], Jgdy[i,j] ]]))
    return e


def myInterpolation(Im, d):
    x = np.arange(0, np.size(Im,1) ,1)
    y = np.arange(0, np.size(Im,0), 1)
    intSpline = scipy.interpolate.RectBivariateSpline(y, x, Im)
    xd = x-d[0]
    yd = y-d[1]
    interpolatedIm = intSpline(yd, xd)
    return interpolatedIm

def myKLTracker(J, I, point, window_size=np.array([41,71]), maxIterations=100):
    dTot = np.zeros((2,1))
    d = np.transpose(np.array([np.Inf, np.Inf]))
    i = 1

    ksize = 5;
    sigma = 1;
    Jg = lowpassFilter(J, ksize, sigma)
    Ig = lowpassFilter(I, ksize, sigma)
    (Jgdx, Jgdy) = image_gradient(J, ksize, sigma)

    JgWarp = Jg
    JgdxWarp = Jgdx
    JgdyWarp = Jgdy
    while i <= maxIterations and np.linalg.norm(d) > 0.1 :
        if i > 1:
            JgWarp = myInterpolation(Jg, dTot)
            JgdxWarp = myInterpolation(Jgdx, dTot)
            JgdyWarp = myInterpolation(Jgdy, dTot)

        (Jgdx2, Jgdy2, Jgdxdy) = getTensorData(JgdxWarp, JgdyWarp)
        T = estimate_T(Jgdx2, Jgdy2, Jgdxdy, point[0], point[1], window_size)
        e = estimate_e(JgWarp, Ig, JgdxWarp, JgdyWarp, point[0], point[1], window_size)
        d = np.linalg.solve(T, -1*e)
        dTot += d
        i += 1
    return dTot


def getTensorImage(im, grad_k_size, grad_sigma, window_size):
    (imdx, imdy) = image_gradient(im, grad_k_size, grad_sigma)
    imShape = np.shape(im);
    gradX2 = np.zeros(imShape)
    gradY2 = np.zeros(imShape)
    gradXY = np.zeros(imShape)
    (Jgdx2, Jgdy2, Jgdxdy) = getTensorData(imdx, imdy)
    count = 1;
    for i in range(0, imShape[0]-1):
        for j in range(0, imShape[1]-1):
            T = estimate_T(Jgdx2, Jgdy2, Jgdxdy, j, i, window_size)
            gradX2[i,j] = T[0,0]
            gradY2[i,j] = T[1,1]
            gradXY[i,j] = T[0,1]
            print("Calculating tensor image: {}".format(count/(imShape[0]*imShape[1]) * 100))
            count += 1
    return (gradX2, gradY2, gradXY)


import matplotlib.image as mpimage
im = mpimage.imread("flower.jpeg")
J = im[:,:,1].astype(float)
dReal = np.transpose(np.array([20.1, 8.9]))
I = myInterpolation(J,dReal)

#d = myKLTracker(J, I, np.array([150,150]))
(gradX2, gradY2, gradXY) = getTensorImage(J, 5, 1, np.array([11,11]))
#print(d)


#plt.figure(1)
#plt.imshow(J)
plt.figure(2)
plt.imshow(gradXY)
plt.show()
