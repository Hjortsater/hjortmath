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
    matrix_zero = _lib.matrix_zero
    matrix_identity = _lib.matrix_identity
    matrix_get_max = _lib.matrix_get_max
    matrix_get_min = _lib.matrix_get_min
    matrix_rows = _lib.matrix_rows
    matrix_cols = _lib.matrix_cols
    matrix_add = _lib.matrix_add
    matrix_add_inplace = _lib.matrix_add_inplace
    matrix_sub = _lib.matrix_sub
    matrix_sub_inplace = _lib.matrix_sub_inplace
    matrix_mul = _lib.matrix_mul
    matrix_mul_inplace = _lib.matrix_mul_inplace
    matrix_seed_random = _lib.matrix_seed_random
    matrix_fill_random = _lib.matrix_fill_random
    matrix_create_from_buffer = _lib.matrix_create_from_buffer
    matrix_inverse = _lib.matrix_inverse
    matrix_determinant = _lib.matrix_determinant
    matrix_log_determinant = _lib.matrix_log_determinant
    matrix_to_list = _lib.matrix_to_list