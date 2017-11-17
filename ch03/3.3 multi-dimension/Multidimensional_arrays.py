# coding: utf-8
import numpy as np

## 1차원 배열
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))  # Return the number of dimensions of an array.
print(A.shape)  # Tuple of array dimensions.
print(A.shape[0])


#
#   1 2
#   3 4
#   5 6
#
## 2차원 배열
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)     # (행,열)   (row, col)




