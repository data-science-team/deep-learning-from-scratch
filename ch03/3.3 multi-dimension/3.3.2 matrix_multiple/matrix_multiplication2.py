# coding: utf-8
import numpy as np

A = np.array([[1,2,3], [4,5,6]])
print(A.shape)

B = np.array([[1,2], [3,4], [5,6]])
print(B.shape)

result = np.dot(A, B)   # Matrix inner product / Matrix multiplication
print(result)

result2 = np.dot(B, A)   # 다른 결과
print(result2)


# 행렬곱에서 행렬수가 안맞는 case
C = np.array([[1,2], [3,4]])
print(A.shape)
print(C.shape)
result3 = np.dot(A, C)
print(result3)





