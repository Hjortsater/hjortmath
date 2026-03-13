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

    @lazy()
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
    
    @lazy()
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
        if not SETTINGS.mutable_eagers:
            return self.__mul__(other)
        if not isinstance(other, (int, float)):
            return self.__mul__(other)
        self._version += 1
        CFunc.matrix_scalar_mul_inplace(self._ptr, float(other), self._ptr, int(SETTINGS.multithreaded))
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

    def evaluate(self) -> Self:
        print("Attempted to evaluate object of type Matrix")
        print("Perhaps the LazyMatrix intended was already evaluated, or the lazy_eval flag is set to dynamic (1)?")
        return self

class LazyMatrix(Matrix):
    class OpEnum(IntEnum):
        ADD = 0
        SUB = 1
        RML = 2
        LML = 3
        SML = 4
        DIV = 5

    __slots__ = ("root", "ops", "operands", "versions","_ptr", "_version")

    def __init__(self, root, ops=None, operands=None, versions=None):
        self.root = root
        self.ops = ops if ops is not None else []
        self.operands = operands if operands is not None else []
        self.versions = versions if versions is not None else []
        self._ptr = root._ptr
        self._version = root._version

    def __del__(self) -> None:
        pass

    def __str__(self) -> str:
        return self.evaluate().__str__()

    def __repr__(self) -> str:
        return self.evaluate().__repr__()

    def __add__(self, other: Union[Matrix, LazyMatrix]) -> LazyMatrix:
        if not isinstance(other, (Matrix, LazyMatrix)):
            return NotImplemented

        new_ops = self.ops + [self.OpEnum.ADD]
        new_operands = self.operands + [other]
        
        if isinstance(other, LazyMatrix):
            new_versions = self.versions + [0]
        else:
            new_versions = self.versions + [other._version]

        return LazyMatrix(
            root=self.root,
            ops=new_ops,
            operands=new_operands,
            versions=new_versions
        )


    def __sub__(self, other: Union[Matrix, LazyMatrix]) -> LazyMatrix:
            if not isinstance(other, (Matrix, LazyMatrix)):
                return NotImplemented

            new_ops = self.ops + [self.OpEnum.SUB]
            new_operands = self.operands + [other]
            
            if isinstance(other, LazyMatrix):
                new_versions = self.versions + [0]
            else:
                new_versions = self.versions + [other._version]

            return LazyMatrix(
                root=self.root,
                ops=new_ops,
                operands=new_operands,
                versions=new_versions
            )


    def __mul__(self, other: Union[Matrix, LazyMatrix, int, float]) -> LazyMatrix:
        if isinstance(other, (int, float)):
            new_ops = self.ops + [self.OpEnum.SML]
            new_operands = self.operands + [float(other)]
            new_versions = self.versions + [0]
        elif isinstance(other, (Matrix, LazyMatrix)):
            new_ops = self.ops + [self.OpEnum.RML]
            new_operands = self.operands + [other]
            new_versions = self.versions + [getattr(other, "_version", 0)]
        else:
            return NotImplemented

        return LazyMatrix(
            root=self.root,
            ops=new_ops,
            operands=new_operands,
            versions=new_versions
        )


    def __rmul__(self, other: Union[Matrix, LazyMatrix, int, float]) -> LazyMatrix:
        if isinstance(other, (int, float)):
            new_ops = self.ops + [self.OpEnum.SML]
            new_operands = self.operands + [float(other)]
            new_versions = self.versions + [0]
        elif isinstance(other, (Matrix, LazyMatrix)):
            new_ops = self.ops + [self.OpEnum.LML]
            new_operands = self.operands + [other]
            new_versions = self.versions + [getattr(other, "_version", 0)]
        else:
            return NotImplemented

        return LazyMatrix(
            root=self.root,
            ops=new_ops,
            operands=new_operands,
            versions=new_versions
        )


    def __truediv__(self, other) -> LazyMatrix:
        if isinstance(other, (int, float)): # NOTE Float scalar division
            return LazyMatrix(self.root, self.ops + [(self.OpEnum.LML, 1.0/float(other))])
        
        if isinstance(other, (Matrix, LazyMatrix)):
            return LazyMatrix(self.root, self.ops + [(self.OpEnum.DIV, other)])        
        return NotImplemented

    def __rtruediv__(self, other) -> LazyMatrix:
        if isinstance(other, (int, float)):
            return (~self) * other
        return NotImplemented

    def __itruediv__(self, other: Union[Matrix, int, float]) -> LazyMatrix:
        return self / other


    @alias("compute")
    def evaluate(self) -> Matrix:

        ops_capsules = []
        for op_type, operand, ver in zip(self.ops, self.operands, self.versions):

            if isinstance(operand, LazyMatrix):
                operand = operand.evaluate()
                ver = getattr(operand, "_version", 0)

            if op_type == self.OpEnum.SML:
                ops_capsules.append((int(op_type), float(operand), ver))
            else:
                ops_capsules.append((int(op_type), operand._ptr, ver))

        result_ptr = CFunc.matrix_evaluate_kernel(self.root._ptr, ops_capsules, int(SETTINGS.multithreaded), int(SETTINGS.simplify))

        return Matrix._init_C_native(result_ptr)


    @alias("inv")
    @property
    def inverse(self) -> Matrix:
        return ~self
    
    @alias("det")
    @property
    def determinant(self) -> float:
        return self.evaluate().determinant