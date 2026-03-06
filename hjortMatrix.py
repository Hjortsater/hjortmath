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
        self.multithreaded: bool = kwargs.get("multithreaded", True)
        self.limit_prints: int = kwargs.get("limit_prints", 0)

    def to_dict(self):
        return vars(self)

SETTINGS = GlobalFlags()

class Matrix:
    __slots__ = ("_ptr", "_flags")
        
    def __init__(self, *rows: Union[int, float, list, tuple], **flags: Any) -> None:
        self._flags = SETTINGS
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
        obj._flags = SETTINGS
        return obj

    def __del__(self) -> None:
        if hasattr(self, "_ptr") and self._ptr: CFunc.matrix_free(self._ptr)
    
    def __repr__(self) -> str:
        from hjort__repr__ import hjort__repr__
        return hjort__repr__(self)

    @lazy()
    def __add__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        result_ptr = CFunc.matrix_create(self.m, self.n)
        CFunc.matrix_add_inplace(self._ptr, other._ptr, result_ptr, int(self._flags.multithreaded))
        return Matrix._init_C_native(result_ptr)
    
    @lazy()
    def __sub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.m != other.m or self.n != other.n: raise ValueError("Dimensions mismatch.")
        new_ptr = CFunc.matrix_sub(self._ptr, other._ptr, int(self._flags.multithreaded))
        return Matrix._init_C_native(new_ptr)

    @lazy()
    def __mul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix): raise NotImplementedError
        if self.n != other.m: raise ValueError("Dimensions mismatch.")
        new_ptr = CFunc.matrix_mul(self._ptr, other._ptr, int(self._flags.multithreaded))
        return Matrix._init_C_native(new_ptr)

    @property
    def m(self) -> int: return CFunc.matrix_rows(self._ptr)
    @property
    def n(self) -> int: return CFunc.matrix_cols(self._ptr)

    @classmethod
    def random(cls, m: int, n: int, **flags: Any) -> Self:
        ptr = CFunc.matrix_create(m, n)
        CFunc.matrix_fill_random(ptr, 0.0, 1.0)
        return cls._init_C_native(ptr)

    @classmethod
    def identity(cls, n: int, **flags: Any) -> Self:
        ptr = CFunc.matrix_identity(n)
        return cls._init_C_native(ptr)
    
    @property
    def inverse(self) -> Matrix:
        ptr = CFunc.matrix_inverse(self._ptr, self._flags.multithreaded)
        return self._init_C_native(ptr)

    @property
    def determinant(self) -> float:
        return CFunc.matrix_determinant(self._ptr, self._flags.multithreaded)

class LazyMatrix(Matrix):
    class OpEnum(IntEnum):
        ADD = 0
        SUB = 1
        MUL = 2
        INV = 3

    def __init__(self, root: Matrix, ops: list = None) -> None:
        self.root = root
        self.ops = ops or []
        self._flags = SETTINGS
        self._ptr = root._ptr

    #NOTE Override the destructor as to not free up pointers before they are evaluated
    def __del__(self) -> None:
        pass
    
    def __repr__(self) -> str:
        ops_ptr = [(op, m._ptr) for op, m in self.ops]
        return f"LazyMatrix(Root: {self.root._ptr}, Ops: {ops_ptr})"
    
    def __add__(self, other: Matrix) -> LazyMatrix:
        self.ops.append((self.OpEnum.ADD, other))
        return self
    
    def __sub__(self, other: Matrix) -> LazyMatrix:
        self.ops.append((self.OpEnum.SUB, other))
        return self
    
    def __mul__(self, other: Matrix) -> LazyMatrix:
        self.ops.append((self.OpEnum.MUL, other))
        return self

    @alias("compute")
    def evaluate(self) -> Matrix:
        def check_dims():
            m, n = self.root.m, self.root.n
            cur_m, cur_n = m, n
            for op, mat in self.ops:
                if op in (self.OpEnum.ADD, self.OpEnum.SUB):
                    if mat.m != cur_m or mat.n != cur_n:
                        raise ValueError("Dimension mismatch.")
                elif op == self.OpEnum.MUL:
                    if cur_n != mat.m:
                        raise ValueError("Dimension mismatch.")
                    cur_n = mat.n
                elif op == self.OpEnum.INV:
                    if mat.m != mat.n:
                        raise ValueError("Inverse requires square matrix.")

        def simplify():
            new_ops = []
            for op, mat in self.ops:
                if op == self.OpEnum.MUL and mat.m == mat.n:
                    if getattr(mat, "is_identity", False):
                        continue
                if new_ops:
                    prev_op, prev_mat = new_ops[-1]
                    if prev_mat._ptr == mat._ptr:
                        if prev_op == self.OpEnum.ADD and op == self.OpEnum.SUB:
                            new_ops.pop()
                            continue
                        if prev_op == self.OpEnum.SUB and op == self.OpEnum.ADD:
                            new_ops.pop()
                            continue
                new_ops.append((op, mat))
            return new_ops

        def compile_ops(ops):
            op_codes = []
            ptrs = []
            for op, mat in ops:
                op_codes.append(int(op))
                ptrs.append(mat._ptr)
            return op_codes, ptrs

        check_dims()
        ops = simplify()
        op_codes, ptrs = compile_ops(ops)

        result_ptr = self.root._ptr
        for op, ptr in zip(op_codes, ptrs):
            if op == self.OpEnum.ADD:
                new_ptr = CFunc.matrix_add(result_ptr, ptr, int(self._flags.multithreaded))
            elif op == self.OpEnum.SUB:
                new_ptr = CFunc.matrix_sub(result_ptr, ptr, int(self._flags.multithreaded))
            elif op == self.OpEnum.MUL:
                new_ptr = CFunc.matrix_mul(result_ptr, ptr, int(self._flags.multithreaded))
            elif op == self.OpEnum.INV:
                new_ptr = CFunc.matrix_inverse(result_ptr, int(self._flags.multithreaded))
            else:
                raise RuntimeError("Unknown operation.")
            result_ptr = new_ptr

        return Matrix._init_C_native(result_ptr)