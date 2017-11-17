# coding: utf-8
import numpy as np

#
#   1
#   2
#
#   1 2
#
# input
X = np.array([1,2])
print(X.shape)

#
#   1 3 5
#   2 4 6
#
# weight
W = np.array([[1,3,5], [2,4,6]])
print(W)
print(W.shape)

Y = np.dot(X,W)
print(Y)

# Y2 = np.dot(W,X)
# print(Y2)