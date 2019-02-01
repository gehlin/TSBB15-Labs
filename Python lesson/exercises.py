import numpy as np
import scipy
from matplotlib import pyplot as plt

sinus_curve = np.sin((np.pi*2) * np.linspace (0 ,1 ,10000))
#plt.plot(sinus_curve,'-.*m')


(x,y) = np.meshgrid(np.arange (-128,129,1),np.arange (-128,129,1))


plt.figure(1)
plt.imshow(x)
plt.figure(2)
plt.imshow(y)
plt.show()
