import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib.image as mpimage
import utilities as ut
import matplotlib

im = mpimage.imread("flower.jpeg")
J = im[:,:,1].astype(float)
dReal = np.transpose(np.array([5.3, 4.1]))
I = ut.myInterpolation(J,dReal)

(ch, x, y) = ut.harris(J,5)

print(x)
print(y)
points = np.array([x, y]).T
oldPoints = points
dTot = np.zeros((2,  np.size(x)))
for i in range(0, np.size(x)-1):
    dTemp = ut.myKLTracker(J, I, points[i,:])
    dTot[0,i] = dTemp[0]
    dTot[1,i] = dTemp[1]
    points[i,0] += dTot[0,i]
    points[i,1] += dTot[1,i]

#p = np.array([[30, 30]]).T
#dTot = ut.myKLTracker(J, I, p)

print(dTot)
