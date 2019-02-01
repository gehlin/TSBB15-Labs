import numpy as np
import scipy
from matplotlib import pyplot as plt

#row_vector = np.array([1,2,3], dtype='float32')
#small_matrix = np.array([[1,2,3], [4,5,6], [7,8,9]], dtype='float32')
#column_vector = np.array([[1,2,3]])
#black_image = np.zeros((64,128,3))

row_vector = np.array([[1,2,3]],  dtype='float32')
small_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]],  dtype='float64')
column_vector = np.array([[1,2,3]]).T
#column_vector = np.transpose(column_vector)
black_image = np.zeros((64,128,3))


# Scalar product
# scalar = row_vector @ column_vector
#print("Scalar: {}".format(scalar))

# Outer product
#matrix = column_vector @ row_vector;
#print("Matrix:\n {}".format(matrix))

# matrix * vector product:
#v1 = small_matrix @ column_vector
#print("matrix*vector:\n {}".format(v1))

# vector * matrix product
#v2 = row_vector @ small_matrix
#print("vector*matrix:\n {}".format(v2))

# Element wise multplication
#em = matrix * small_matrix
#print("Element wise:\n {}".format(em))

#plt.imshow(black_image)


# Change the red channel of the top most pixel to one
black_image[0,0,0] = 1;
# Change all color channels of the pixel beside it to one, setting it to white
black_image[0,1,:] = 1;
# Change to bottom right square of the image to be blue:
black_image[-1:-10:-1,-1:-10:-1,1] = 1

#plt.imshow(black_image)
#plt.show()

# 4. FANCY SYNTAX
things_list = ['a', 'b', 1, 2, 3, 4, 1.2, 1.3, 1.4]
numbers_list = [item for item in things_list if isinstance(item, (int, float))]
letters_list = [item for item in things_list if isinstance(item, str)]
print("======")
for number in numbers_list:
    print(number)
print("======")
for idx, number in enumerate(letters_list):
    print("Letter number: {}, is {}".format(idx,letters_list))

def loop_function(N):
    """ Build a list using a loop """
    l = []
    for n in range(N):
        l.append(n)
    return n

def listcomprehension_function(N):
    """ Build a list using a list comprehension """
    return [n for n in range(0,N)]
