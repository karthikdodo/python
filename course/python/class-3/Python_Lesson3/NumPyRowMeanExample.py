import numpy as np

X = np.random.rand(3, 2)
print(X)


Y = X - X.mean(axis=1).reshape(-1,1)

print(Y)