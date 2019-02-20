import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib.image as mpimage
import utilities as ut
import lab1
from matplotlib.patches import Circle
import matplotlib.axes as axes


im1 = lab1.load_lab_image("chessboard_1.png")
imShape = np.shape(im1)

images = np.zeros((imShape[0], imShape[1], 10))
for i in range(1,11):
    filename = "chessboard_" + str(i) + ".png"
    images[:,:,i-1] = lab1.load_lab_image(filename)

window_size = np.array([21,21])
harrisWindow = np.array([3,3])
maxIterations = 100
ksize = 5
sigma = 1
dThresh = 0.05
kappa = 0.05
harrisThresh = 0.1


(ch, x, y) = ut.harris(images[:,:,0], 5, ksize, sigma, harrisWindow, kappa, harrisThresh)
dTot = np.zeros((2,  np.size(x), 9))
points = np.zeros((2,  np.size(x), 10))
points[:,:,0] = np.array([x,y])

for j in range(0,9):
    fig, ax = plt.subplots()
    ax.imshow(images[:,:,j])

    for i in range(0, np.size(x)):
        dTemp = ut.myKLTracker(images[:,:,j], images[:,:,j+1], points[:,i,j], window_size, maxIterations, ksize, sigma, dThresh)
        dTot[0,i,j] = dTemp[0]
        dTot[1,i,j] = dTemp[1]
        points[0,i,j+1] = points[0,i,j] + dTot[0,i,j]
        points[1,i,j+1] = points[1,i,j] + dTot[1,i,j]
        circ = Circle(points[:,i,j], 4, facecolor = 'None', linewidth = 1.0, edgecolor = 'r')
        ax.add_patch(circ)
    print("Image: " + str(j+1))
    #print("Estimated d for image " + str(j+1) + " to " + str(j+2))
    #print(dTot[:,:,j])
    #print("Estimated point in image " + str(j+1))
    #print(np.round(points[:,1,j]))

plt.show()
