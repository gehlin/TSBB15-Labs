import numpy as np
import scipy
from matplotlib import pyplot as plt



image_filename = 'flower.jpeg'
# using pillow
from PIL import Image
im1 = np.array(Image.open(image_filename))
print(im1.shape)

# using opencv
import cv2
im2 = cv2.imread(image_filename)
print(im2.shape)

# using  scikit -image
import skimage.io as skio
#im3 = skio.imread(image_filename)
#print(im3.shape)

# using matplotlib
import matplotlib.image as mpimage
im4 = mpimage.imread(image_filename)
print(im4.shape)

plt.figure(1)
plt.imshow(im1)
plt.show()

plt.figure(2)
plt.imshow(im2)
plt.show()

plt.figure(3)
plt.imshow(im4)
plt.show()
