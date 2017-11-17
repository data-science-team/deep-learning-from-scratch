import numpy as np

A = np.array([[1,2], [3,4]])
print(A.shape)

B = np.array([[5,6], [7,8]])
print(B.shape)

result = np.dot(A, B)   # Matrix inner product / Matrix multiplication
print(result)

result2 = np.dot(B, A)   # 다른 결과
print(result2)