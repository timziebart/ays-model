#!/usr/bin/env python3

import ctypes as ct
import numpy as np

DIM = 3
NUM = 4
#point_t = 
array_t = (ct.c_double * DIM) * NUM

libviability = ct.cdll.LoadLibrary("./libviability.so")
libviability.first_test.argtypes = ()
#doublepp = np.ctypeslib.ndpointer(
#        dtype=np.uintp, 
#        ndim=1, 
#        flags="C_CONTIGUOUS"
#        )
libviability.add_point.argtypes = (ct.c_void_p, )
libviability.add_points.argtypes = (
        ct.c_void_p,
        ct.c_uint64,
        ct.c_uint32, 
        ct.POINTER(array_t)
#        ct.POINTER((ct.c_double * DIM))
#        doublepp
        )
libviability.add_points.restype = None
libviability.access_list.restype = ct.c_double
libviability.access_2d_list.restype = ct.c_double

result = libviability.first_test()
print(result)

tree = libviability.create_tree()
libviability.add_point(tree)

points = np.array([
    [ 0, 1, 1],
    [ 2, 3, 4],
    [ 4, 2, 1],
    [ 5, 2, 1],
    ],
    dtype=np.float64)
#points = np.ascontiguousarray(points)

number = 20
mylist = np.linspace(0, 10, number, endpoint=False, dtype=np.float64)

get = lambda arr, ind: libviability.access_list(ct.c_ulong(ind), arr.ctypes.data_as(ct.POINTER(ct.c_double)))
get_2d = lambda arr, ind1, ind2: libviability.access_2d_list(
        ct.c_ulong(ind1), 
        ct.c_ulong(ind2), 
        arr.ctypes.data_as(ct.POINTER(ct.c_double))
        )

#for ind in range(number):
#    print(float(get(mylist, ind)))

for ind in range(points.size):
    print(float(get(points, ind)))
print("bla")
for ind1 in range(points.shape[0]):
    for ind2 in range(points.shape[1]):
        print(float(get_2d(points, ind1, ind2)), end=" ")
    print()


#pointspp = (points.__array_interface__['data'][0] 
#              + np.arange(points.shape[0])*points.strides[0]).astype(np.uintp) 

#points = array_t()
#for d in range(DIM):
#    for n in range(NUM):
#        points[n][d] = d + 0.1*n
#
#print(points)

#num, dim = points.shape
#assert dim == 3
#
#num = ct.c_int64(num)
#dim = ct.c_int32(dim)
#print(num, dim)

#print("size", NUM*DIM)
#print("shape", (NUM, DIM))
#libviability.add_points(tree, NUM, DIM, ct.byref(points))

