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
        self.lazy_eval: int = kwargs.get("lazy_eval", 1)
        self.sig_digits: int = kwargs.get("sig_digits", 3)
        self.use_color: bool = kwargs.get("use_color", True)
        self.suppress_zeroes: str = kwargs.get("suppress_zeroes", " ")
        self.multithreaded: bool = kwargs.get("multithreaded", True)
        self.limit_prints: int = kwargs.get("limit_prints", 0)

    def to_dict(self):
        return vars(self)

SETTINGS = GlobalFlags()

class Matrix:
    __slots__ = ("_ptr",)
        
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

    @classmethod
    def _init_C_native(cls, ptr: int) -> Self:
        if not ptr: raise MemoryError("Null pointer from C backend.")
        obj = cls.__new__(cls)
        obj._ptr = ptr
        return obj

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr:
            CFunc.matrix_free(self._ptr)
    
    def __repr__(self) -> str:
        from hjort__repr__ import hjort__repr__
        return hjort__repr__(self)

    @lazy()
    def __add__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        result_ptr = CFunc.matrix_add(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        return Matrix._init_C_native(result_ptr)
    
    def __iadd__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        CFunc.matrix_add_inplace(self._ptr, other._ptr, self._ptr, int(SETTINGS.multithreaded))
        return self
    
    @lazy()
    def __sub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        new_ptr = CFunc.matrix_sub(self._ptr, other._ptr, int(SETTINGS.multithreaded))
        return Matrix._init_C_native(new_ptr)
    
    def __isub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        CFunc.matrix_sub_inplace(self._ptr, other._ptr, self._ptr, int(SETTINGS.multithreaded))
        return self

    @lazy()
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
    
    def __imul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.n != other.m:
            raise ValueError("Dimensions mismatch.")
        CFunc.matrix_mul_inplace(self._ptr, other._ptr, self._ptr, int(SETTINGS.multithreaded))
        return self
    
    @lazy()
    def __truediv__(self, other: Union[Matrix, int, float]) -> Matrix:
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        if isinstance(other, Matrix):
            result_ptr = CFunc.matrix_solve(self._ptr, other._ptr)
        if isinstance(other, LazyMatrix):
            other = other.evaluate()
            result_ptr = CFunc.matrix_solve(self._ptr, other._ptr)
        if not result_ptr:
            raise ValueError("Singular matrix in inverse right multiplication")
        return Matrix._init_C_native(result_ptr)



    @property
    def m(self) -> int: return CFunc.matrix_rows(self._ptr)
    @property
    def n(self) -> int: return CFunc.matrix_cols(self._ptr)

    @classmethod
    def random(cls, m: int, n: int) -> Self:
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


class LazyMatrix(Matrix):
    class OpEnum(IntEnum):
        ADD = 0
        SUB = 1
        RML = 2
        LML = 3

    def __init__(self, root: Matrix, ops: list = None) -> None:
        self.root = root
        self.ops = ops or []
        self._ptr = root._ptr

    def __del__(self) -> None:
        pass

    def __repr__(self) -> str:
        return repr(self.evaluate())

    def __add__(self, other) -> LazyMatrix:
        return LazyMatrix(self.root, self.ops + [(self.OpEnum.ADD, other)])

    def __sub__(self, other) -> LazyMatrix:
        return LazyMatrix(self.root, self.ops + [(self.OpEnum.SUB, other)])

    def __mul__(self, other) -> LazyMatrix:
        if isinstance(other, (int, float, Matrix)):
            return LazyMatrix(self.root, self.ops + [(self.OpEnum.RML, other)])

    def __rmul__(self, other) -> LazyMatrix:
        if isinstance(other, (int, float, Matrix)):
            return LazyMatrix(self.root, self.ops + [(self.OpEnum.LML, other)])

    def __truediv__(self, other) -> LazyMatrix:
        if isinstance(other, (int, float)): # NOTE Float scalar division
            return LazyMatrix(self.root, self.ops + [(self.OpEnum.LML, 1.0/float(other))])
        
        if not isinstance(other, (Matrix, LazyMatrix)): # NOTE Inverse matrix right mul
            pass

    def __rtruediv__(self, other) -> LazyMatrix:
        if isinstance(other, (int, float)):
            return (~self) * other
        return NotImplemented

    def __itruediv__(self, other: Union[Matrix, int, float]) -> LazyMatrix:
        return self / other


    @alias("compute")
    def evaluate(self) -> Matrix:
        simplified_ops = []
        for op_type, operand in self.ops:
            if isinstance(operand, LazyMatrix):
                operand = operand.evaluate()
            
            if simplified_ops:
                prev_type, prev_op = simplified_ops[-1]
                if isinstance(operand, Matrix) and isinstance(prev_op, Matrix):
                    if operand._ptr == prev_op._ptr:
                        is_add_sub = (op_type == self.OpEnum.ADD and prev_type == self.OpEnum.SUB)
                        is_sub_add = (op_type == self.OpEnum.SUB and prev_type == self.OpEnum.ADD)
                        if is_add_sub or is_sub_add:
                            simplified_ops.pop()
                            continue
            
            simplified_ops.append((op_type, operand))
        
        res_ptr = CFunc.matrix_clone(self.root._ptr)
        mt = int(SETTINGS.multithreaded)

        for op_type, op in simplified_ops:
            if op_type == self.OpEnum.ADD:
                CFunc.matrix_add_inplace(res_ptr, op._ptr, res_ptr, mt)
            elif op_type == self.OpEnum.SUB:
                CFunc.matrix_sub_inplace(res_ptr, op._ptr, res_ptr, mt)
            elif op_type == self.OpEnum.RML:
                if isinstance(op, (int, float)):
                    CFunc.matrix_scalar_mul_inplace(res_ptr, float(op), res_ptr)
                else:
                    CFunc.matrix_mul_inplace(res_ptr, op._ptr, res_ptr, mt)
            elif op_type == self.OpEnum.LML:
                if isinstance(op, (int, float)):
                    CFunc.matrix_scalar_mul_inplace(res_ptr, float(op), res_ptr)
                else:
                    new_ptr = CFunc.matrix_mul(op._ptr, res_ptr, mt)
                    CFunc.matrix_free(res_ptr)
                    res_ptr = new_ptr

        return Matrix._init_C_native(res_ptr)

    @alias("inv")
    @property
    def inverse(self) -> Matrix:
        return ~self
    
    @alias("det")
    @property
    def determinant(self) -> float:
        return self.evaluate().determinant