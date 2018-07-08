import numpy as np
array = np.random.rand(4,4)
print(array)

mat = np.mat(array)
print(mat)

invmat = mat.I
print(invmat)

danweiMat = np.eye(4)
print(danweiMat)

i = mat*invmat
print(i)
