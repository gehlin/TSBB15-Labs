import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import convolve2d  as conv2
import os

################################################################################
################################################################################
def lowpassFilter(Im, ksize, sigma):
    (x,y) = np.meshgrid(np.arange(-(ksize-1)/2,(ksize-1)/2+1,1),np.arange(-(ksize-1)/2,(ksize-1)/2+1,1))
    g = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
    Img = conv2(Im, g, mode='same')
    return Img

################################################################################
################################################################################
def image_gradient(Im, ksize, sigma):
    (x,y) = np.meshgrid(np.arange(-(ksize-1)/2,(ksize-1)/2+1,1),np.arange(-(ksize-1)/2,(ksize-1)/2+1,1))
    g = 1/(2*np.pi*sigma**2)*np.exp(-(x**2+y**2)/(2*sigma**2))
    gdx = -(x/sigma**2)*g
    gdy = -(y/sigma**2)*g
    Imdx = conv2(Im, gdx, mode='same')
    Imdy = conv2(Im, gdy, mode='same')
    return (Imdx, Imdy)

################################################################################
################################################################################
def estimate_T(Jgdx, Jgdy, x, y, window_size):
    T = np.zeros((2,2))
    xMax = np.size(Jgdx,1)-1
    yMax = np.size(Jgdy,0)-1

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
            T = T + np.array([ [Jgdx[i,j]**2, Jgdx[i,j]*Jgdy[i,j] ], [ Jgdx[i,j]*Jgdy[i,j], Jgdy[i,j]**2 ]])
    return T

################################################################################
################################################################################
def estimate_e(Jg, Ig, Jgdx, Jgdy, x, y, window_size):
    e = np.zeros((2,1))
    xMax = np.size(Jgdx,1)-1
    yMax = np.size(Jgdy,0)-1
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
            e = e + (Ig[i,j]-Jg[i,j])*np.transpose(np.array([[ Jgdx[i,j], Jgdy[i,j] ]]))
    return e

################################################################################
################################################################################
def myInterpolation(Im, d):
    x = np.arange(0, np.size(Im,1) ,1)
    y = np.arange(0, np.size(Im,0), 1)
    intSpline = scipy.interpolate.RectBivariateSpline(y, x, Im)
    xd = x-d[0]
    yd = y-d[1]
    interpolatedIm = intSpline(yd, xd)
    return interpolatedIm

################################################################################
################################################################################
def myKLTracker(J, I, point, window_size=np.array([15,15]), maxIterations=100, ksize=5, sigma=1, dThresh=0.05):
    dTot = np.zeros((2,1))
    d = np.transpose(np.array([np.Inf, np.Inf]))
    i = 1

    Jg = lowpassFilter(J, ksize, sigma)
    Ig = lowpassFilter(I, ksize, sigma)
    (Jgdx, Jgdy) = image_gradient(J, ksize, sigma)

    JgWarp = Jg
    JgdxWarp = Jgdx
    JgdyWarp = Jgdy
    while i <= maxIterations and np.linalg.norm(d) > dThresh :
        if i > 1:
            JgWarp = myInterpolation(Jg, dTot)
            JgdxWarp = myInterpolation(Jgdx, dTot)
            JgdyWarp = myInterpolation(Jgdy, dTot)

        T = estimate_T(JgdxWarp, JgdyWarp, point[0], point[1], window_size)
        e = estimate_e(JgWarp, Ig, JgdxWarp, JgdyWarp, point[0], point[1], window_size)
        d = np.linalg.solve(T, -1*e)
        dTot += d
        i += 1
    return dTot

################################################################################
################################################################################
def getTensorImage(im, grad_k_size, grad_sigma, window_size=np.array([3,3])):
    (imdx, imdy) = image_gradient(im, grad_k_size, grad_sigma)
    imShape = np.shape(im);
    gradX2 = np.zeros(imShape)
    gradY2 = np.zeros(imShape)
    gradXY = np.zeros(imShape)
    count = 1;
    for i in range(0, imShape[0]-1):
        for j in range(0, imShape[1]-1):
            T = estimate_T(imdx, imdy, j, i, window_size)
            gradX2[i,j] = T[0,0]
            gradY2[i,j] = T[1,1]
            gradXY[i,j] = T[0,1]
            count += 1
        print("Calculating tensor image: {}".format(count/(imShape[0]*imShape[1]) * 100))
    os.system('clear')
    return (gradX2, gradY2, gradXY)

################################################################################
################################################################################
def harris(im, K, grad_k_size=5, grad_sigma=1, window_size=np.array([3,3]), kappa=0.05, threshold=0.1):
    (gradX2, gradY2, gradXY) = getTensorImage(im, grad_k_size, grad_sigma, window_size)
    detT = np.multiply(gradX2,gradY2) - np.multiply(gradXY, gradXY);
    traceT = gradX2+gradY2
    ch = detT-kappa*np.multiply(traceT, traceT)
    ch = ch/np.amax(ch)
    chMax = scipy.signal.order_filter(ch, np.ones((5,5)), 25-1)
    ch[ch != chMax] = 0
    chMax[chMax < threshold] = 0

    width = np.size(im,1)
    height = np.size(im,0)
    edgeThreshold = 10;
    ch[0 : edgeThreshold, :]=0
    ch[:, 0 : edgeThreshold]=0
    ch[:, width-edgeThreshold : width] =0
    ch[height-edgeThreshold : height, :] =0

    (x,y) = getLargest(ch,K)
    return (ch,x,y)

################################################################################
################################################################################
def getLargest(im, K):
    imShape = np.shape(im);

    im = im.ravel()

    sortedIdx = np.flip(np.argsort(im),0)
    largestIdx = sortedIdx[0:K]

    y = largestIdx // imShape[1]
    x = largestIdx % imShape[1]
    return (x,y)

#def drawCircles(im, x, y):
#    fig,ax = plt.figure()
#    for i in range(0, np.size(x)-1):
#        circ = matplotlib.patches.Circle(np.array([[x[i], y[i]]]),3)
#        ax.add_patch(circ)
#    plt.show()
