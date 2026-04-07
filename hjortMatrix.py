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
        if not SETTINGS.mutable_eagers or self._ptr == other._ptr:
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
        HAD = 6

    class Node:
        __slots__ = ()
        def simplify(self):
            return self

    class Leaf(Node):
        __slots__ = ("ptr", "version", "scalar", "is_scalar")
        def __init__(self, ptr=None, version=0, scalar=None, is_scalar=False):
            self.ptr = ptr
            self.version = version
            self.scalar = scalar
            self.is_scalar = is_scalar

    class BinOp(Node):
        __slots__ = ("op", "left", "right")
        def __init__(self, op, left, right):
            self.op = op
            self.left = left
            self.right = right

        def simplify(self):
            l = self.left.simplify()
            r = self.right.simplify()

            if self.op == LazyMatrix.OpEnum.ADD:
                # Check if both operands are the same matrix
                if isinstance(l, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.Leaf):
                    if not l.is_scalar and not r.is_scalar and l.ptr == r.ptr and l.version == r.version and l.ptr is not None:
                        return LazyMatrix.BinOp(
                            LazyMatrix.OpEnum.SML,
                            LazyMatrix.Leaf(scalar=2.0, is_scalar=True),
                            l
                        )
                
                # Check if left is scalar mul and right is the same matrix
                if isinstance(l, LazyMatrix.BinOp) and l.op == LazyMatrix.OpEnum.SML and isinstance(l.right, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.Leaf):
                    if not r.is_scalar and l.right.ptr == r.ptr and l.right.version == r.version and r.ptr is not None:
                        return LazyMatrix.BinOp(
                            LazyMatrix.OpEnum.SML,
                            LazyMatrix.Leaf(scalar=l.left.scalar + 1.0, is_scalar=True),
                            r
                        )
                
                # Check if right is scalar mul and left is the same matrix
                if isinstance(r, LazyMatrix.BinOp) and r.op == LazyMatrix.OpEnum.SML and isinstance(r.right, LazyMatrix.Leaf) and isinstance(l, LazyMatrix.Leaf):
                    if not l.is_scalar and r.right.ptr == l.ptr and r.right.version == l.version and l.ptr is not None:
                        return LazyMatrix.BinOp(
                            LazyMatrix.OpEnum.SML,
                            LazyMatrix.Leaf(scalar=r.left.scalar + 1.0, is_scalar=True),
                            l
                        )
                
                if isinstance(l, LazyMatrix.BinOp) and isinstance(r, LazyMatrix.BinOp):
                    if l.op == LazyMatrix.OpEnum.SML and r.op == LazyMatrix.OpEnum.SML:
                        if isinstance(l.right, LazyMatrix.Leaf) and isinstance(r.right, LazyMatrix.Leaf):
                            if l.right.ptr == r.right.ptr and l.right.version == r.right.version and l.right.ptr is not None:
                                return LazyMatrix.BinOp(
                                    LazyMatrix.OpEnum.SML,
                                    LazyMatrix.Leaf(scalar=l.left.scalar + r.left.scalar, is_scalar=True),
                                    l.right
                                )
                
                if isinstance(l, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.BinOp):
                    if r.op == LazyMatrix.OpEnum.SML and isinstance(r.right, LazyMatrix.Leaf):
                        if l.ptr == r.right.ptr and l.version == r.right.version and l.ptr is not None:
                            return LazyMatrix.BinOp(
                                LazyMatrix.OpEnum.SML,
                                LazyMatrix.Leaf(scalar=1.0 + r.left.scalar, is_scalar=True),
                                l
                            )

            if self.op == LazyMatrix.OpEnum.SUB:
                # Check if both operands are the same matrix: A - A = 0*A
                if isinstance(l, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.Leaf):
                    if not l.is_scalar and not r.is_scalar and l.ptr == r.ptr and l.version == r.version and l.ptr is not None:
                        return LazyMatrix.BinOp(
                            LazyMatrix.OpEnum.SML,
                            LazyMatrix.Leaf(scalar=0.0, is_scalar=True),
                            l
                        )
                
                # Check if left is scalar mul and right is the same matrix: k*A - A = (k-1)*A
                if isinstance(l, LazyMatrix.BinOp) and l.op == LazyMatrix.OpEnum.SML and isinstance(l.right, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.Leaf):
                    if not r.is_scalar and l.right.ptr == r.ptr and l.right.version == r.version and r.ptr is not None:
                        return LazyMatrix.BinOp(
                            LazyMatrix.OpEnum.SML,
                            LazyMatrix.Leaf(scalar=l.left.scalar - 1.0, is_scalar=True),
                            r
                        )
                
                # Check if right is scalar mul and left is the same matrix: A - k*A = (1-k)*A
                if isinstance(r, LazyMatrix.BinOp) and r.op == LazyMatrix.OpEnum.SML and isinstance(r.right, LazyMatrix.Leaf) and isinstance(l, LazyMatrix.Leaf):
                    if not l.is_scalar and r.right.ptr == l.ptr and r.right.version == l.version and l.ptr is not None:
                        return LazyMatrix.BinOp(
                            LazyMatrix.OpEnum.SML,
                            LazyMatrix.Leaf(scalar=1.0 - r.left.scalar, is_scalar=True),
                            l
                        )
                
                if isinstance(l, LazyMatrix.BinOp) and isinstance(r, LazyMatrix.BinOp):
                    if l.op == LazyMatrix.OpEnum.SML and r.op == LazyMatrix.OpEnum.SML:
                        if isinstance(l.right, LazyMatrix.Leaf) and isinstance(r.right, LazyMatrix.Leaf):
                            if l.right.ptr == r.right.ptr and l.right.version == r.right.version and l.right.ptr is not None:
                                return LazyMatrix.BinOp(
                                    LazyMatrix.OpEnum.SML,
                                    LazyMatrix.Leaf(scalar=l.left.scalar - r.left.scalar, is_scalar=True),
                                    l.right
                                )
                
                if isinstance(l, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.BinOp):
                    if r.op == LazyMatrix.OpEnum.SML and isinstance(r.right, LazyMatrix.Leaf):
                        if l.ptr == r.right.ptr and l.version == r.right.version and l.ptr is not None:
                            return LazyMatrix.BinOp(
                                LazyMatrix.OpEnum.SML,
                                LazyMatrix.Leaf(scalar=1.0 - r.left.scalar, is_scalar=True),
                                l
                            )

                # New: (X + Y) - Y = X
                if isinstance(l, LazyMatrix.BinOp) and l.op == LazyMatrix.OpEnum.ADD:
                    if isinstance(l.right, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.Leaf):
                        if l.right.ptr == r.ptr and l.right.version == r.version and r.ptr is not None:
                            return l.left
                    if isinstance(l.left, LazyMatrix.Leaf) and isinstance(r, LazyMatrix.Leaf):
                        if l.left.ptr == r.ptr and l.left.version == r.version and r.ptr is not None:
                            return l.right

            return LazyMatrix.BinOp(self.op, l, r)

    __slots__ = ("tree", "_ptr", "_version")

    def __init__(self, tree_or_matrix):
        if isinstance(tree_or_matrix, (Matrix, LazyMatrix)):
            self.tree = self._to_node(tree_or_matrix)
        else:
            self.tree = tree_or_matrix
        
        if isinstance(self.tree, self.Leaf) and not self.tree.is_scalar:
            self._ptr = self.tree.ptr
            self._version = self.tree.version
        else:
            self._ptr = 0
            self._version = 0

    def _to_node(self, obj):
        if isinstance(obj, LazyMatrix):
            return obj.tree
        if isinstance(obj, Matrix):
            return self.Leaf(ptr=obj._ptr, version=obj._version)
        if isinstance(obj, (int, float)):
            return self.Leaf(scalar=float(obj), is_scalar=True)
        return obj

    def __add__(self, other):
        result = LazyMatrix(self.BinOp(self.OpEnum.ADD, self.tree, self._to_node(other)))
        if SETTINGS.lazy_eval == 0:
            return result.evaluate()
        return result

    def __sub__(self, other):
        result = LazyMatrix(self.BinOp(self.OpEnum.SUB, self.tree, self._to_node(other)))
        if SETTINGS.lazy_eval == 0:
            return result.evaluate()
        return result

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = LazyMatrix(self.BinOp(self.OpEnum.SML, self._to_node(other), self.tree))
        else:
            result = LazyMatrix(self.BinOp(self.OpEnum.RML, self.tree, self._to_node(other)))
        if SETTINGS.lazy_eval == 0:
            return result.evaluate()
        return result

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            result = LazyMatrix(self.BinOp(self.OpEnum.SML, self._to_node(other), self.tree))
        else:
            result = LazyMatrix(self.BinOp(self.OpEnum.LML, self._to_node(other), self.tree))
        if SETTINGS.lazy_eval == 0:
            return result.evaluate()
        return result

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = LazyMatrix(self * (1.0 / other))
        else:
            result = LazyMatrix(self.BinOp(self.OpEnum.DIV, self.tree, self._to_node(other)))
        if SETTINGS.lazy_eval == 0:
            return result.evaluate()
        return result

    def __matmul__(self, other):
        result = LazyMatrix(self.BinOp(self.OpEnum.HAD, self.tree, self._to_node(other)))
        if SETTINGS.lazy_eval == 0:
            return result.evaluate()
        return result

    def __iadd__(self, other):
        return self + other

    def evaluate(self):
        stree = self.tree
        ops_stack = []

        def build(node):
            if isinstance(node, self.Leaf):
                if node.is_scalar:
                    ops_stack.append((int(self.OpEnum.SML), node.scalar, 0))
                else:
                    ops_stack.append((-1, node.ptr, node.version))
                return

            build(node.left)
            build(node.right)
            ops_stack.append((int(node.op), None, 0))

        build(stree)
        result_ptr = CFunc.matrix_evaluate_stack_kernel(
            None,
            ops_stack,
            int(SETTINGS.multithreaded),
            int(SETTINGS.simplify)
        )
        return Matrix._init_C_native(result_ptr)

    def __str__(self):
        return self.evaluate().__str__()

    def __repr__(self):
        return self.evaluate().__repr__()

    @property
    def m(self):
        if self._ptr: return CFunc.matrix_rows(self._ptr)
        return self.evaluate().m

    @property
    def n(self):
        if self._ptr: return CFunc.matrix_cols(self._ptr)
        return self.evaluate().n

    def to_numpy(self):
        return self.evaluate().to_numpy()