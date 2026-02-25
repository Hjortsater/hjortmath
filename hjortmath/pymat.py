from .imports import *
from . import cmat
from .customdecorators import alias, validate_dimensions, performance_warning

class Matrix:
    """
    CUSTOM MATRIX CLASS IN PYTHON WITH A C BACKEND FOR EXPENSIVE COMPUTATIONS.

    WRITTEN FOR FUN, NOT FOR SPEED!
    """
    
    # --- INITIALIZATION ---

    def __init__(self, *rows: Any, **kwargs: Any) -> None:
        """CONSTRUCTOR FOR MATRIX"""
        self.entries: List[float] = []
        self.m: int = 0
        self.n: int = 0

        self.use_C: bool = kwargs.get('use_C', True)
        self.force_C: bool = kwargs.get('force_C', False)
        self.use_color: bool = kwargs.get('use_color', True)
        self.sig_digits: int = kwargs.get('sig_digits', 4)
        self.disable_perf_hints: bool = kwargs.get('disable_warnings', False)
        
        self._cached_repr: Optional[str] = None

        def parse_single_row(rows_data: Tuple[Any, ...]) -> Optional[Tuple[List[float], int, int]]:
            """PARSE A SINGLE DIMENSION INPUT"""
            if all(isinstance(i, (int, float)) for i in rows_data):
                return [float(i) for i in rows_data], 1, len(rows_data)
            return None

        def parse_multi_row(rows_data: Tuple[Any, ...]) -> Optional[Tuple[List[float], int, int]]:
            """PARSE MULTIPLE ROW INPUT"""
            if all(isinstance(row, tuple) and len(row) == len(rows_data[0]) for row in rows_data):
                entries: List[float] = [float(i) for row in rows_data for i in row]
                return entries, len(rows_data), len(rows_data[0])
            return None

        if not rows:
            raise ValueError("Matrix cannot be empty.")

        result: Optional[Tuple[List[float], int, int]] = parse_single_row(rows) or parse_multi_row(rows)
        if result is None:
            raise TypeError("Each row must be a tuple of equal length.")
        
        self.entries, self.m, self.n = result

    # --- INTERNAL HELPERS ---

    def _to_tuple_form(self) -> List[Tuple[float, ...]]:
        """CONVERT INTERNAL FLAT LIST TO LIST OF TUPLES"""
        return Matrix.to_tuple_form(self.entries, self.n, self.m)

    def _smul(self, other: float) -> Self:
        """PERFORM SCALAR MULTIPLICATION"""
        return Matrix._from_flat([other * i for i in self.entries], self.n, self.m)

    def _determinant(self, _internal: bool = False) -> float:
        """INTERNAL DETERMINANT CALCULATION LOGIC"""
        if self.n == 1:
            return self.entries[0]
        elif self.n == 2:
            return self.entries[0] * self.entries[3] - self.entries[1] * self.entries[2]

        if self.use_C:
            return float(cmat.mat_det(self.entries, self.n))

        if not _internal:
            yellow_bold: str = "\033[1;33m"
            reset: str = "\033[0m"
            print(f"{yellow_bold}Just a heads up!{reset} "
                f"The python implementation of determinant uses Laplace expansion, which is very slow for large matrices (O(n!)). Consider using the C implementation for better performance. Set use_C=False to force Python, or disable_perf_hints=True to turn off this warning.")

        def laplace_expansion(entries: List[float], n: int) -> float:
            """RECURSIVE LAPLACE EXPANSION FOR DETERMINANT"""
            if n == 2:
                return entries[0] * entries[3] - entries[1] * entries[2]
            det: float = 0.0
            for j in range(n):
                minor_entries: List[float] = []
                for row in range(1, n):
                    for col in range(n):
                        if col == j:
                            continue
                        minor_entries.append(entries[row * n + col])
                sign: int = 1 if j % 2 == 0 else -1
                element: float = entries[j]
                det += sign * element * laplace_expansion(minor_entries, n - 1)
            return det

        return laplace_expansion(self.entries, self.n)

    # --- STATIC & CLASS METHODS ---

    @staticmethod
    def to_tuple_form(lst: List[float], n: int, m: int) -> List[Tuple[float, ...]]:
        """CONVERT A FLAT LIST TO A NESTED TUPLE STRUCTURE"""
        return [tuple(lst[i * n : (i + 1) * n]) for i in range(m)]

    @classmethod
    def _from_flat(cls, entries: List[float], n: int, m: int) -> Self:
        """CREATE MATRIX INSTANCE FROM FLATTENED LIST"""
        tuples: List[Tuple[float, ...]] = cls.to_tuple_form(entries, n, m)
        return cls(*tuples)

    @alias("ident", "IDENT", "I")
    @classmethod
    def identity(cls, n: int) -> Self:
        """CREATE AN IDENTITY MATRIX OF SIZE N"""
        if n <= 0:
            raise ValueError(f"Provided matrix dimension (n={n}) must be greater than 0")
        
        entries: List[float] = [
            1.0 if i == j else 0.0
            for i in range(n)
            for j in range(n)
        ]
        return cls(*cls.to_tuple_form(entries, n, n))
    
    @alias("zero", "ZERO")
    @classmethod
    def zero_matrix(cls, n: int) -> Self:
        """CREATE A ZERO MATRIX OF SIZE N"""
        if n <= 0:
            raise ValueError(f"Provided matrix dimension (n={n}) must be greater than 0")
        return cls(*cls.to_tuple_form([0.0] * (n * n), n, n))

    @alias("rand", "RAND", "R")
    @classmethod
    def random(cls, n: int, m: int, low: float = 0.0, high: float = 1.0) -> Self:
        """CREATE A RANDOM MATRIX OF SIZE NXm"""
        if n <= 0 or m <= 0:
            raise ValueError(f"Matrix dimensions must be positive (got {n}x{m})")
        if low > high:
            raise ValueError(f"Low bound {low} cannot be greater than high bound {high}")
        
        entries: List[float] = [random.uniform(low, high) for _ in range(n * m)]
        return cls(*cls.to_tuple_form(entries, n, m))

    # --- PROPERTIES ---

    @alias("T")
    @property
    def transpose(self) -> Self:
        """RETURN TRANSPOSED MATRIX"""
        transposed_entries: List[float] = []
        for j in range(self.n):
            for i in range(self.m):
                transposed_entries.append(self.entries[i * self.n + j])
        return Matrix(*self.to_tuple_form(transposed_entries, self.m, self.n))

    @alias("det")
    @property
    @validate_dimensions("square")
    def determinant(self) -> float:
        """CALCULATE THE DETERMINANT"""
        return self._determinant(_internal=False)

    @alias("inv", "INV")
    @property
    @validate_dimensions("square")
    def inverse(self) -> Self:
        """CALCULATE THE INVERSE MATRIX"""
        det: float = self._determinant(_internal=True)
        if abs(det) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted.")

        if self.n == 1:
            return Matrix((1.0 / self.entries[0],))
        elif self.n == 2:
            inv_entries: List[float] = [
                self.entries[3] / det,
                -self.entries[1] / det,
                -self.entries[2] / det,
                self.entries[0] / det
            ]
            return Matrix(*self.to_tuple_form(inv_entries, 2, 2))

        if self.use_C:
            c_inv: List[float] = cmat.mat_inv(self.entries, self.n)
            return Matrix(*self.to_tuple_form(c_inv, self.n, self.n))

        raise NotImplementedError("Inverse for matrices larger than 2x2 is not implemented in pure Python.")

    # --- DUNDER METHODS ---

    def __repr__(self) -> str:
        """GENERATE STRING REPRESENTATION OF MATRIX"""
        if self._cached_repr is not None:
            return self._cached_repr

        if not self.entries:
            return "[]"

        formatted: List[str]
        if hasattr(self, 'sig_digits') and self.sig_digits >= 0:
            formatted = [f"{val:.{self.sig_digits}g}" for val in self.entries]
        else:
            formatted = [f"{val}" for val in self.entries]

        min_val: float = min(self.entries)
        max_val: float = max(self.entries)
        range_val: float = max_val - min_val

        def get_color(value: float) -> str:
            """GET ANSI COLOR CODE BASED ON VALUE INTENSITY"""
            if range_val == 0:
                return "\033[0m"
            normalized: float = (value - min_val) / range_val
            normalized = max(0.0, min(1.0, normalized))
            colors: List[int] = [82, 118, 226, 214, 196]
            color_idx: int = int(normalized * (len(colors) - 1))
            return f"\033[38;5;{colors[color_idx]}m"

        width: int = max(len(f) for f in formatted)
        reset: str = "\033[0m"
        rows_list: List[str] = []

        if len(self.entries) <= self.n:
            row_parts: List[str] = [
                f"{get_color(val)}{display:>{width}}{reset}" if getattr(self, 'use_color', False) else f"{display:>{width}}"
                for val, display in zip(self.entries, formatted)
            ]
            row: str = '  '.join(row_parts)
            self._cached_repr = f"[ {row} ]"
            return self._cached_repr

        for j in range(0, len(self.entries), self.n):
            row_parts_multi: List[str] = []
            for idx, val in enumerate(self.entries[j : j + self.n]):
                display_val: str = formatted[j + idx]
                if getattr(self, 'use_color', False):
                    row_parts_multi.append(f"{get_color(val)}{display_val:>{width}}{reset}")
                else:
                    row_parts_multi.append(f"{display_val:>{width}}")

            row_str: str = '  '.join(row_parts_multi)
            if j == 0:
                rows_list.append(f"┌ {row_str} ┐")
            elif j + self.n >= len(self.entries):
                rows_list.append(f"└ {row_str} ┘")
            else:
                rows_list.append(f"│ {row_str} │")

        self._cached_repr = '\n'.join(rows_list)
        return self._cached_repr

    @validate_dimensions("elementwise")
    @performance_warning()
    def __add__(self, other: Self) -> Self:
        """ADD TWO MATRICES"""
        if not self.force_C:
            summed_entries: List[float] = [i + j for i, j in zip(self.entries, other.entries)]
            return Matrix._from_flat(summed_entries, self.n, self.m)
        
        C_entries: List[float] = cmat.mat_add(self.entries, other.entries, self.m, self.n)
        return Matrix._from_flat(C_entries, self.n, self.m)

    @validate_dimensions("elementwise")
    @performance_warning()
    def __sub__(self, other: Self) -> Self:
        """SUBTRACT TWO MATRICES"""
        if not self.force_C:
            subbed_entries: List[float] = [i - j for i, j in zip(self.entries, other.entries)]
            return Matrix._from_flat(subbed_entries, self.n, self.m)
        
        C_entries: List[float] = cmat.mat_sub(self.entries, other.entries, self.m, self.n)
        return Matrix._from_flat(C_entries, self.n, self.m)

    @validate_dimensions("matmul")
    @performance_warning()
    def __mul__(self, other: Union[Self, float, int]) -> Union[Self, float]:
        """PERFORM MATRIX MULTIPLICATION OR SCALAR MULTIPLICATION"""
        if isinstance(other, (float, int)):
            return self._smul(float(other))

        if not self.use_C:
            mult_entries: List[float] = []
            for i in range(self.m):
                for j in range(other.n):
                    val: float = sum(self.entries[i * self.n + k] * other.entries[k * other.n + j] for k in range(self.n))
                    mult_entries.append(val)
            return Matrix._from_flat(mult_entries, other.n, self.m)

        C_result: List[float] = cmat.mat_mul(self.entries, other.entries, self.m, self.n, other.n)
        
        if len(C_result) == 1 and self.m == 1 and other.n == 1:
            return float(C_result[0])
            
        return Matrix._from_flat(C_result, other.n, self.m)

    @validate_dimensions("elementwise")
    @performance_warning()
    def __matmul__(self, other: Union[Self, float, int]) -> Self:
        """PERFORM HADAMARD PRODUCT (ELEMENT-WISE MULTIPLICATION)"""
        if isinstance(other, (float, int)):
            return self._smul(float(other))
        
        if not self.force_C:
            mult_entries: List[float] = [i * j for i, j in zip(self.entries, other.entries)]
            return Matrix._from_flat(mult_entries, self.n, self.m)
            
        C_entries: List[float] = cmat.hadamard(self.entries, other.entries, self.m, self.n)
        return Matrix._from_flat(C_entries, self.n, self.m)

