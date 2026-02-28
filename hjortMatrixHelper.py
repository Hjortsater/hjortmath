import hjortMatrixWrapper as _lib
"""

File which handles ctypes boilte_plater and exposes C-functions in a public CFunc class.

Written by Erik Hjortsäter February 27th 2026.

"""

# Name wrapper
class CFunc:
    matrix_create = _lib.matrix_create
    matrix_free = _lib.matrix_free
    matrix_set = _lib.matrix_set
    matrix_get = _lib.matrix_get
    matrix_fill = _lib.matrix_fill
    matrix_get_max = _lib.matrix_get_max
    matrix_get_min = _lib.matrix_get_min
    matrix_rows = _lib.matrix_rows
    matrix_cols = _lib.matrix_cols
    matrix_add = _lib.matrix_add
    matrix_seed_random = _lib.matrix_seed_random
    matrix_fill_random = _lib.matrix_fill_random