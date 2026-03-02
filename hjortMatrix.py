from __future__ import annotations
from typing import Any, Optional, Union, Tuple, List
from hjortMatrixHelper import CFunc

"""

Simple linear algebra class written for fun, for experience, for ease of use and hopefully actual implementations down the line!
This version of the code boasts a more prominent C-backend, instead of only off-loading heavy operations via ctypes.
    *This way multithreading should scale
    *This way type-conversion overhead should be reduced
    
Written by Erik Hjortsäter February 27th 2026.

"""

class Matrix:

    ### DUNDER METHODS (and then some)

    class _Flags:
        def __init__(self, **kwargs: Any) -> None:
            """Constructor for the flags helper"""
            self.sig_digits: int = kwargs.get("sig_digits", 3)
            self.use_color: bool = kwargs.get("use_color", True)
            self.multithreaded: bool = kwargs.get("multithreaded", True)
            self.limit_prints: int = kwargs.get("limit_prints", 0)

    def __init__(self, *rows: Union[int, float, list, tuple], **flags: Any) -> None:
        """Pythonic constructor, slow but Python-native"""
        self._flags: Matrix._Flags = self._Flags(**flags)
        flat: list[float] = list(rows)

        if not len(rows):
            raise ValueError("Matrix cannot be empty.")

        # NOTE 1D vector input
        if all(isinstance(i, (int, float)) for i in rows):
            m, n = 1, len(rows)
            flat = list(rows)

        # NOTE 2D matrix input (can still result in vector)
        elif all(isinstance(r, (list, tuple)) for r in rows):
            m = len(rows)
            n = len(rows[0])

            if any(len(r) != n for r in rows):
                raise ValueError("Rows must be same length.")

            flat = [float(val) for r in rows for val in r]

        else:
            raise TypeError("Invalid constructor input.")
        
        import array
        buffer: array.array = array.array('d', flat)
        ptr: Optional[int] = CFunc.matrix_create_from_buffer(buffer, m, n)
        
        if not ptr:
            raise MemoryError("C backend allocation failed")
        
        self._ptr: int = ptr

      
    @classmethod
    def __init__C_native(cls, ptr: int, **flags: Any) -> Self:
        """
        C-idiomatic constructor, takes a C pointer and optional flags.
        """
        if not ptr:
            raise MemoryError("Null pointer from C backend.")

        obj: Self = cls.__new__(cls)      # bypass Python __init__
        obj._ptr = ptr
        obj._flags = cls._Flags(**flags)
        return obj

    __slots__ = ("_ptr", "_flags")

    def __del__(self) -> None:
        """Object destuctor, calls back-end freeing"""
        if hasattr(self, "_ptr") and self._ptr:
                CFunc.matrix_free(self._ptr)
    
    def __repr__(self) -> str:
        """Object representation. Lengthy method, seperated by file."""
        from hjort__repr__ import hjort__repr__
        return hjort__repr__(self)


    ### DUNDER OPERATIONS

    def __add__(self, other: Self) -> Self:
        if not isinstance(other, Matrix):
            raise NotImplementedError

        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition.")

        result_ptr: Optional[int] = CFunc.matrix_create(self.m, self.n)
        if not result_ptr:
            raise MemoryError("C backend failed to allocate result matrix.")

        CFunc.matrix_add_inplace(self._ptr, other._ptr, result_ptr, int(self._flags.multithreaded))

        return Matrix.__init__C_native(
            result_ptr,
            sig_digits=self._flags.sig_digits,
            use_color=self._flags.use_color,
            multithreaded=self._flags.multithreaded
        )
    
    def __iadd__(self, other: Self) -> Self:
        if not isinstance(other, Matrix):
            raise NotImplementedError
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition.")
        if not CFunc.matrix_add_inplace(
            self._ptr,
            other._ptr,
            self._ptr,
            int(self._flags.multithreaded)
        ):
            raise RuntimeError("C backend failed in inplace addition.")
        return self
            
    def __sub__(self, other: Self) -> Self:
        """Subtract two matrices using the C backend and return a new Matrix."""
        if not isinstance(other, Matrix):
            raise NotImplementedError

        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for addition.")

        new_ptr: Optional[int] = CFunc.matrix_sub(self._ptr, other._ptr, int(self._flags.multithreaded))
        if not new_ptr:
            raise MemoryError("C backend failed to allocate result matrix.")

        return Matrix.__init__C_native(
            new_ptr,
            sig_digits=self._flags.sig_digits,
            use_color=self._flags.use_color,
            multithreaded=self._flags.multithreaded
        )

    def __isub__(self, other: Self) -> Self:
        if not isinstance(other, Matrix):
            raise NotImplementedError
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrix dimensions must match for subtraction.")
        if not CFunc.matrix_sub_inplace(
            self._ptr,
            other._ptr,
            self._ptr,
            int(self._flags.multithreaded)
        ):
            raise RuntimeError("C backend failed in inplace subtraction.")
        return self

    def __mul__(self, other: Self) -> Self:
        """Multiply two matrices using the C backend and return a new Matrix."""
        if not isinstance(other, Matrix):
            raise NotImplementedError

        if self.n != other.m:
            raise ValueError("Matrix dimensions must match for multiplication.")

        new_ptr: Optional[int] = CFunc.matrix_mul(self._ptr, other._ptr, int(self._flags.multithreaded))
        if not new_ptr:
            raise MemoryError("C backend failed to allocate result matrix.")
        
        return Matrix.__init__C_native(
            new_ptr,
            sig_digits=self._flags.sig_digits,
            use_color=self._flags.use_color,
            multithreaded=self._flags.multithreaded
        )
        

    ### PROPERTIES

    @property
    def m(self) -> int:
        return CFunc.matrix_rows(self._ptr)

    @property
    def n(self) -> int:
        return CFunc.matrix_cols(self._ptr)


    ### MISCELLANEOUS

    @classmethod
    def random(
        cls,
        m: int,
        n: int,
        min_val: float = 0.0,
        max_val: float = 1.0,
        **flags: Any
    ) -> Self:
        """Create a random matrix using the C backend"""
        ptr: Optional[int] = CFunc.matrix_create(m, n)
        if not ptr:
            raise MemoryError("Failed to allocate matrix in C backend.")
        
        CFunc.matrix_fill_random(ptr, min_val, max_val)

        return cls.__init__C_native(ptr, **flags)
    
    @property
    def determinant(self) -> float:
        if self.n and self.m:
            return CFunc.matrix_determinant(self._ptr)