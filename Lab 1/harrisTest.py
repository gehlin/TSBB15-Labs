import numpy as np
import scipy
from matplotlib import pyplot as plt
import matplotlib.image as mpimage
import utilities as ut
import lab1

J = lab1.load_lab_image("chessboard_1.png")
#I, J, dTrue = lab1.get_cameraman() #dTrue = (1, -2)
(ch, x, y) = ut.harris(J,5)

print(x)
print(y)

plt.figure()
plt.imshow(J)
plt.figure()
plt.imshow(ch)
plt.show()
