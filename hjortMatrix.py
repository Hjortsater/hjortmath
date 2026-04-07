from __future__ import annotations
from typing import Any, Optional, Union, Self
from hjortMatrixHelper import CFunc
from hjortDecorators import *
from enum import IntEnum

"""

Simple linear algebra class written for fun, for experience, for ease of use and hopefully actual implementations down the line!
This version of the code boasts a more prominent C-backend, instead of only off-loading heavy operations via ctypes.
    *This way multithreading should scale
    *This way type-conversion overhead should be reduced
    
Written by Erik Hjortsäter February 27th 2026.

"""

class GlobalFlags:
    def __init__(self, **kwargs: Any) -> None:
        self.mutable_eagers: bool = kwargs.get("mutable_eagers", False)
        self.simplify: bool = kwargs.get("simplify", True)
        self.sig_digits: int = kwargs.get("sig_digits", 6)
        self.use_color: bool = kwargs.get("use_color", True)
        self.suppress_zeroes: str = kwargs.get("suppress_zeroes", " ")
        self.multithreaded: bool = kwargs.get("multithreaded", True)
        self.limit_prints: int = kwargs.get("limit_prints", 0)

    def to_dict(self):
        return vars(self)

SETTINGS = GlobalFlags()

class Matrix:
    __slots__ = ("_ptr", "_version")
        
    def __init__(self, *rows: Union[int, float, list, tuple]) -> None:
        flat: list[float] = list(rows)
        if not len(rows): raise ValueError("Matrix cannot be empty.")
        if all(isinstance(i, (int, float)) for i in rows):
            m, n = 1, len(rows)
            flat = list(rows)
        elif all(isinstance(r, (list, tuple)) for r in rows):
            m = len(rows)
            n = len(rows[0])
            if any(len(r) != n for r in rows): raise ValueError("Rows must be same length.")
            flat = [float(val) for r in rows for val in r]
        else: raise TypeError("Invalid constructor input.")
        import array
        buffer = array.array('d', flat)
        ptr = CFunc.matrix_create_from_buffer(buffer, m, n)
        if not ptr: raise MemoryError("C backend allocation failed")
        self._ptr = ptr
        self._version = 0

    @classmethod
    def _init_C_native(cls, ptr: int) -> Self:
        if not ptr: raise MemoryError("Null pointer from C backend.")
        obj = cls.__new__(cls)
        obj._ptr = ptr
        obj._version = 0
        return obj

    def __del__(self) -> None:
        return
    
    def __str__(self) -> str:
        from hjort_str_ import hjort_str_
        return hjort_str_(self)
    
    def __repr__(self) -> str:
        from hjort_str_ import hjort_str_
        return hjort_str_(self)

    def __add__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        result_ptr = CFunc.matrix_add(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        return Matrix._init_C_native(result_ptr)
    
    def __iadd__(self, other: Matrix) -> Matrix:
        if not SETTINGS.mutable_eagers:
            return self.__add__(other)
        if not isinstance(other, Matrix): raise NotImplementedError
        self._version += 1
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        CFunc.matrix_add_inplace(self._ptr, other._ptr, self._ptr, int(SETTINGS.multithreaded))
        return self
    
    def __sub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        new_ptr = CFunc.matrix_sub(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        return Matrix._init_C_native(new_ptr)
    
    def __isub__(self, other: Matrix) -> Matrix:
        if not SETTINGS.mutable_eagers:
            return self.__sub__(other)
        if not isinstance(other, Matrix): raise NotImplementedError
        self._version += 1
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        CFunc.matrix_sub_inplace(self._ptr, other._ptr, self._ptr, int(SETTINGS.multithreaded))
        return self

    def __mul__(self, other: Union[Matrix, int, float]) -> Matrix:
        if isinstance(other, (int, float)):
            new_ptr = CFunc.matrix_scalar_mul(self._ptr, float(other), int(SETTINGS.multithreaded))
            return Matrix._init_C_native(new_ptr)
        if not isinstance(other, Matrix):
            raise NotImplementedError
        if self.n != other.m:
            raise ValueError("Dimensions mismatch.")
        new_ptr = CFunc.matrix_mul(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        return Matrix._init_C_native(new_ptr)

    def __rmul__(self, other: Union[int, float]) -> Matrix:
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented
    
    def __matmul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch for elementwise mul.")
        result_ptr = CFunc.matrix_elementwise_mul(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        return Matrix._init_C_native(result_ptr)
    
    def __imul__(self, other: Matrix) -> Matrix:
        if not SETTINGS.mutable_eagers:
            return self.__mul__(other)
        if not isinstance(other, (int, float)):
            return self.__mul__(other)
        self._version += 1
        CFunc.matrix_scalar_mul_inplace(self._ptr, float(other), self._ptr, int(SETTINGS.multithreaded))
        return self
    
    def __truediv__(self, other: Union[Matrix, int, float]) -> Matrix:
        """Right inverse: A / B solves A @ X = B for X."""
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        if isinstance(other, Matrix):
            # Solve A @ X = B for X
            result_ptr = CFunc.matrix_solve(self._ptr, other._ptr)
            if not result_ptr:
                raise ValueError("Singular matrix in division")
            return Matrix._init_C_native(result_ptr)
        raise NotImplementedError
    
    def __floordiv__(self, other: Matrix) -> Matrix:
        """Left inverse: A // B solves B @ X = A for X."""
        if not isinstance(other, Matrix):
            raise NotImplementedError
        # Solve B @ X = A for X
        result_ptr = CFunc.matrix_solve(other._ptr, self._ptr)
        if not result_ptr:
            raise ValueError("Singular matrix in left division")
        return Matrix._init_C_native(result_ptr)

    @property
    def m(self) -> int: return CFunc.matrix_rows(self._ptr)
    @property
    def n(self) -> int: return CFunc.matrix_cols(self._ptr)

    @classmethod
    def random(cls, m: int, n: int) -> Self:
        if not m or not n:
            raise ValueError("Matrix dimensions cannot be zero: {m}x{n}")
        ptr = CFunc.matrix_random(m, n, 0.0, 1.0)
        return cls._init_C_native(ptr)

    @alias("I")
    @classmethod
    def identity(cls, n: int) -> Self:
        ptr = CFunc.matrix_identity(n)
        return cls._init_C_native(ptr)
    
    @alias("inv")
    @property
    def inverse(self) -> Matrix:
        ptr = CFunc.matrix_inverse(self._ptr, SETTINGS.multithreaded)
        return self._init_C_native(ptr)

    @alias("det")
    @property
    def determinant(self) -> float:
        return CFunc.matrix_determinant(self._ptr, SETTINGS.multithreaded)
    
    @alias("ldet", "logdet")
    @property
    def log_determinant(self) -> float:
        """Log determinant. Returns signed log|det(A)|."""
        return CFunc.matrix_log_determinant(self._ptr)
    
    def solve(self, other: Matrix) -> Matrix:
        """Solve A @ X = other for X. Equivalent to other / A."""
        if not isinstance(other, Matrix):
            raise TypeError("solve() expects a Matrix")
        result_ptr = CFunc.matrix_solve(other._ptr, self._ptr)
        if not result_ptr:
            raise ValueError("Singular matrix in solve")
        return Matrix._init_C_native(result_ptr)
    
    def solve_lstsq(self, other: Matrix) -> Matrix:
        """Least squares solve: minimize ||A @ X - other||_2."""
        if not isinstance(other, Matrix):
            raise TypeError("solve_lstsq() expects a Matrix")
        result_ptr = CFunc.matrix_solve_lstsq(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        if not result_ptr:
            raise ValueError("Least squares solve failed")
        return Matrix._init_C_native(result_ptr)
    
    def solve_spd(self, other: Matrix) -> Matrix:
        """Solve for symmetric positive definite A: A @ X = other."""
        if not isinstance(other, Matrix):
            raise TypeError("solve_spd() expects a Matrix")
        result_ptr = CFunc.matrix_solve_spd(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        if not result_ptr:
            raise ValueError("Solve SPD failed (matrix may not be SPD or singular)")
        return Matrix._init_C_native(result_ptr)

    def to_list(self):
        return CFunc.matrix_to_list(self._ptr)

    def to_numpy(self):
        import numpy as np
        return np.array(self.to_list())

    @classmethod
    def from_list(cls, lst):
        return cls._init_C_native(CFunc.matrix_from_list(lst))

    @classmethod
    def from_numpy(cls, arr):
        return cls.from_list(arr.tolist())

