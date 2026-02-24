"""
Python wrapper for libcmat.so

Provides matrix operations backed by C for speed.
Matrices are flattened row-major lists of doubles.

All functions return a new Python list.
"""

import ctypes
import os
from customdecorators import *


_lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libcmat.so"))





_lib.mat_add.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # A
    ctypes.POINTER(ctypes.c_double),  # B
    ctypes.POINTER(ctypes.c_double),  # C (output)
    ctypes.c_size_t                   # size
]
_lib.mat_add.restype = None


_lib.mat_sub.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
_lib.mat_sub.restype = None


_lib.hadamard.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
_lib.hadamard.restype = None

_lib.mat_mul.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
_lib.mat_mul.restype = None


_lib.scalar_mul.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t
]
_lib.scalar_mul.restype = None


_lib.mat_det.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.c_size_t
]
_lib.mat_det.restype = ctypes.c_double

_lib.mat_inv.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int
]
_lib.mat_inv.restype = None

@alias("Help")
class Helpers():
    @staticmethod
    def _to_c_array(py_list):
        """Convert Python list to ctypes array"""
        size = len(py_list)
        return (ctypes.c_double * size)(*py_list)

    @staticmethod
    def _new_c_array(size):
        """Create empty ctypes array"""
        return (ctypes.c_double * size)()

    @staticmethod
    def _to_py_list(c_array, size):
        """Convert ctypes array to Python list"""
        return [c_array[i] for i in range(size)]



def mat_add(A, B, m=None, n=None):
    size = len(A)

    if len(B) != size:
        raise ValueError("Arrays must have same length")

    A_arr = Help._to_c_array(A)
    B_arr = Help._to_c_array(B)
    C_arr = Help._new_c_array(size)

    _lib.mat_add(A_arr, B_arr, C_arr, size)

    return Help._to_py_list(C_arr, size)


def mat_sub(A, B, m=None, n=None):
    size = len(A)

    if len(B) != size:
        raise ValueError("Arrays must have same length")

    A_arr = Help._to_c_array(A)
    B_arr = Help._to_c_array(B)
    C_arr = Help._new_c_array(size)

    _lib.mat_sub(A_arr, B_arr, C_arr, size)

    return Help._to_py_list(C_arr, size)


def hadamard(A, B, m=None, n=None):
    size = len(A)

    if len(B) != size:
        raise ValueError("Arrays must have same length")

    A_arr = Help._to_c_array(A)
    B_arr = Help._to_c_array(B)
    C_arr = Help._new_c_array(size)

    _lib.hadamard(A_arr, B_arr, C_arr, size)

    return Help._to_py_list(C_arr, size)

def mat_mul(A, B, m, n, p):

    A_arr = Help._to_c_array(A)
    B_arr = Help._to_c_array(B)
    C_arr = Help._new_c_array(m*p)

    _lib.mat_mul(A_arr, B_arr, C_arr, m, n, p)
    return Help._to_py_list(C_arr, m*p)

def scalar_mul(A, scalar, m=None, n=None):
    size = len(A)

    A_arr = Help._to_c_array(A)
    C_arr = Help._new_c_array(size)

    _lib.scalar_mul(A_arr, ctypes.c_double(scalar), C_arr, size)

    return Help._to_py_list(C_arr, size)


def mat_det(A, n):
    if len(A) != n * n:
        raise ValueError("Matrix list size does not match provided dimensions.")

    A_arr = Help._to_c_array(A)
    
    result = _lib.mat_det(A_arr, n)
    
    return float(result)

def mat_inv(A, n):
    if len(A) != n * n:
        raise ValueError("Matrix list size does not match provided dimensions.")

    A_arr = Help._to_c_array(A)
    C_arr = Help._new_c_array(n * n)

    _lib.mat_inv(A_arr, C_arr, ctypes.c_int(n))

    return Help._to_py_list(C_arr, n * n)