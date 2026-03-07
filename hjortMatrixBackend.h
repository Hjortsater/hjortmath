// hjortMatrixBackend.h
#ifndef HJORTMATRIXBACKEND_H
#define HJORTMATRIXBACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <omp.h>

// Minimal Matrix struct
typedef struct Matrix {
    int m;
    int n;
    double* data;
} Matrix;

#define MAT(M, i, j) ((M)->data[(i) * (M)->n + (j)])

// Core allocation/deallocation
Matrix* matrix_create(int m, int n);
Matrix* matrix_create_from_buffer(size_t m, size_t n, const double* data);
Matrix* matrix_clone(const Matrix* src);
void matrix_copy(const Matrix* src, Matrix* dst);
void matrix_free(Matrix* M);

// Accessors
int matrix_rows(Matrix* M);
int matrix_cols(Matrix* M);
void matrix_set(Matrix* M, int i, int j, double value);
double matrix_get(Matrix* M, int i, int j);

// Utilities
Matrix* matrix_zero(int m, int n);
Matrix* matrix_identity(int n);
Matrix* matrix_random(int m, int n, double min, double max);
void matrix_fill(Matrix* M, double value);

// Math operations
Matrix* matrix_add(const Matrix* A, const Matrix* B, int multithreaded);
Matrix* matrix_add_inplace(const Matrix* A, const Matrix* B, Matrix* C, int multithreaded);
Matrix* matrix_sub(const Matrix* A, const Matrix* B, int multithreaded);
Matrix* matrix_sub_inplace(const Matrix* A, const Matrix* B, Matrix* C, int multithreaded);
Matrix* matrix_mul(Matrix* A, Matrix* B, int multithreaded);
Matrix* matrix_mul_inplace(Matrix* A, Matrix* B, Matrix* C, int multithreaded);
Matrix* matrix_scalar_mul(Matrix* A, double scalar, int multithreaded);
int matrix_scalar_mul_inplace(Matrix* A, Matrix* C, double scalar, int multithreaded);

// Linear algebra
Matrix* matrix_inverse(const Matrix* A, int multithreaded);
int matrix_inverse_inplace(const Matrix* A, Matrix* C, int multithreaded);
Matrix* matrix_solve(const Matrix* A, const Matrix* B);
Matrix* matrix_solve_spd(const Matrix* A, const Matrix* B, int multithreaded);
int matrix_factor_lu(Matrix* A, int* ipiv, int multithreaded);
Matrix* matrix_solve_factored(const Matrix* A_factored, const int* ipiv, const Matrix* B, int multithreaded);
Matrix* matrix_solve_lstsq(const Matrix* A, const Matrix* B, int multithreaded);
double matrix_determinant(const Matrix* M, int multithreaded);
double matrix_log_determinant(const Matrix* M, int* sign);

// Max/min
double matrix_get_max(Matrix* M);
double matrix_get_min(Matrix* M);

#ifdef __cplusplus
}
#endif

#endif