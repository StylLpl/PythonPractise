import numpy as np
i

A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

C = A * B 

print(C)

#create a null vector of size 10
print("create a null vector Z of size 10")
Z = np.zeros(10)
print(Z)

#change the fifth values to 1
print("change the fifth values to 1")
Z[4] = 1
print(Z)

#create a vector with values ranging from 10 to 49
print("create a vector E with values ranging from 10 to 49")
E = np.arange(10, 50)
print(E)

#reverse a vector E
print("reverse a vector E")
E = E[::-1]
print(E)

#create a 3x3 matrix M with values ranging from 0 to 8
M = np.arange(9).reshape(3, 3)
print("create a 3x3 matrix M with values ranging from 0 to 8")
print(M)

print("find indicies of non-zero elements from [1, 2, 0, 0, 4, 0]")
nz = np.nonzero([1, 2, 0, 0, 4, 0])
print(nz)

print("create a 3x3 identity matrix")
im = np.eye(3, order="C")
print(im)

print("create 3x3x3 array with random values")
A = np.random.random((3, 3, 3))
print(A)

print("find minimum and maximum of A")
B = np.random.random((2, 2))
print(B)
print(B.min(), B.max())

print("create a 2d array with 1 on the border and 0 inside")
C = np.ones((10, 10))
C[1:-1, 1:-1] = 0
print(C)

print(0.3 == 3 *0.1)

print("create a 5x5 matrix with values 1, 2, 3, 4 just below the diagonal")
D = np.diag(1+np.arange(4), k=-1) #try to k=-2, k=0 to know more
print(D)

print("consider a (6, 7, 8) shape array, what is the in dex (x, y, z) of the 100th element")
print(np.unravel_index(100, (6, 7, 8)))

print("create a 8x8 matrix and fill it with a checkerboard pattern")
F = np.zeros((8, 8), dtype=int)
F[1::2, ::2] = 1
F[::2, 1::2] = 1
print(F)

print("create a checkerboard 8x8 matrix using the tile function")
Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(Z)


#create a custom dtype that describes a color as four unsighed bytes (RGBA)
color = np.dtype([  ("r", np.ubyte, 1),
                    ("g", np.ubyte, 1), 
                    ("b", np.ubyte, 1), 
                    ("a", np.ubyte, 1)])

print("multiply a 5x3 matrix by a 3x2 matrix (real matrix product)")
Z = np.dot(np.ones((5, 3)), np.ones((3, 2)))
print(Z)

#negative all elements which are between 3 and 8, in place
print("negative all elements which are between 3 and 8, in place")
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1

print(np.array(0) / np.array(0.))

#lam tron so (how to round away from zero a float array)
Z = np.random.uniform(-10, +10, 10) #from -10 to +10 have 10 elements
print(np.trunc(Z + np.copysign(0.5, Z)))

print("extract the integer part of a random array using 5 different methods")
Z = np.random.uniform(0, 10, 10)
print(Z)
print("print(Z - Z%1)")
print(Z - Z%1)
print("np.floor(Z)")
print(np.floor(Z))
print("print(np.ceil(Z) - 1)")
print(np.ceil(Z) - 1)
print("print(Z.astype(int))")
print(Z.astype(int))
print("print(np.trunc(Z))")
print(np.trunc(Z))


def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(), dtype=float, count=-1)
print(Z)

print("create a 5x5 matrix row values ranging from 0 to 4")
Z = np.zeros((5, 5))
Z += np.arange(5)
print(Z)




