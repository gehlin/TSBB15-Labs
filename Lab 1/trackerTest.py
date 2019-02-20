import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib.image as mpimage
import utilities as ut
import lab1


I, J, dTrue = lab1.get_cameraman() #dTrue = (1, -2)

window = np.array([41, 71])
point = np.array([[120, 85]]).T
dEst = ut.myKLTracker(I, J, point, window, maxIterations=100)

print(dEst)
print(dTrue)
