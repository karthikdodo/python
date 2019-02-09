import numpy as np

a=np.array([1,2,3,4,5,6,7,8,9])

print(a)

b = a.reshape((3,3))

print(b)

c=b * 10 + 4

print(c)

Af = np.array([1, 2, 3], float)

print(Af)

print(a.dtype)

print(Af.dtype)

# as a range of values

d=np.arange(10)
print(d)

# By Specifying the number of elements
e=np.linspace(0,1,5)
print(e)

f=np.zeros((2,2))
print(f)

f=np.ones((2,2))
print(f)

g=np.empty((1,3))
print(g)

g=np.eye(3)
print(g)

g=np.diag([1,2,3,4])
print(g)


print(np.sum(a))

h=a.sum()
print(h)

h=np.random.random((2,3))
print(h)

h=np.random.normal(loc=1.0, scale=2.0, size=(2,2))
print(h)

np.savetxt("h_out.txt", h)
